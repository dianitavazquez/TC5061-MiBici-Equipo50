"""Feature and target construction for the station-hour P2 classifier."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici_ml.config import (
    CLUSTERING_FEATURES_FILE,
    CLUSTERING_MIN_ACTIVE_DAYS,
    CLUSTERING_STATIONS_CLEAN_FILE,
    CLUSTERING_TRIPS_FILE,
    P1_FEATURES_FILE,
    P1_TRIPS_FILE,
)
from mibici_ml.dataset import TRIP_COLUMNS, read_csv_compatible

app = typer.Typer(no_args_is_help=True)

TIMESTAMP_COL = "Inicio_del_viaje"
STATION_COL = "Origen_Id"
P3_TIME_RANGES = {
    "manana": range(6, 10),
    "mediodia": range(10, 16),
    "tarde": range(16, 20),
    "noche": [*range(20, 24), *range(0, 6)],
}
P3_HOUR_TO_RANGE = {
    hour: time_range for time_range, hours in P3_TIME_RANGES.items() for hour in hours
}


def load_clean_trips(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load only operational columns and apply the documented quality rules."""
    frames = []
    for path in sorted(raw_dir.glob("datos_abiertos_*.csv")):
        frame = read_csv_compatible(path, usecols=TRIP_COLUMNS)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No hay archivos datos_abiertos_*.csv en {raw_dir}")

    trips = pd.concat(frames, ignore_index=True)
    raw_rows = len(trips)
    trips["Inicio_del_viaje"] = pd.to_datetime(trips["Inicio_del_viaje"], errors="coerce")
    trips["Fin_del_viaje"] = pd.to_datetime(trips["Fin_del_viaje"], errors="coerce")
    valid = trips[TRIP_COLUMNS].notna().all(axis=1)
    valid &= trips["Fin_del_viaje"].ge(trips["Inicio_del_viaje"])
    valid &= trips["Origen_Id"].ne(trips["Destino_Id"])
    trips = trips.loc[valid].copy()
    trips["Origen_Id"] = trips["Origen_Id"].astype("int32")
    trips["Destino_Id"] = trips["Destino_Id"].astype("int32")
    return trips, {"raw_trips": raw_rows, "clean_trips": len(trips)}


def _hourly_event_summary(trips: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ordered events and calculate intra-hour maximum imbalance."""
    departures = trips[["Inicio_del_viaje", "Origen_Id"]].rename(
        columns={"Inicio_del_viaje": "event_time", "Origen_Id": "station_id"}
    )
    departures = departures.assign(delta=1, departures=1, arrivals=0)
    arrivals = trips[["Fin_del_viaje", "Destino_Id"]].rename(
        columns={"Fin_del_viaje": "event_time", "Destino_Id": "station_id"}
    )
    arrivals = arrivals.assign(delta=-1, departures=0, arrivals=1)
    events = pd.concat([departures, arrivals], ignore_index=True)

    # Combining simultaneous events first makes the result independent of row order.
    events = (
        events.groupby(["station_id", "event_time"], as_index=False, sort=False)[
            ["delta", "departures", "arrivals"]
        ]
        .sum()
        .sort_values(["station_id", "event_time"])
    )
    events["hour"] = events["event_time"].dt.floor("h")
    events["cumulative_imbalance"] = events.groupby(["station_id", "hour"], sort=False)[
        "delta"
    ].cumsum()
    summary = events.groupby(["station_id", "hour"], sort=False).agg(
        departures=("departures", "sum"),
        arrivals=("arrivals", "sum"),
        max_bike_imbalance=("cumulative_imbalance", "max"),
        min_bike_imbalance=("cumulative_imbalance", "min"),
    )
    return summary.reset_index()


def build_station_hour_dataset(
    trips: pd.DataFrame,
    stations: pd.DataFrame,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Build one-hour-ahead features and two demand-pressure proxy labels."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha debe estar en el intervalo (0, 1].")

    station_table = stations[["id", "dpcapacity"]].rename(
        columns={"id": "station_id", "dpcapacity": "capacity"}
    )
    station_table = station_table.dropna().query("capacity > 0").copy()
    station_table["station_id"] = station_table["station_id"].astype("int32")
    valid_ids = set(station_table["station_id"])
    trips = trips[trips["Origen_Id"].isin(valid_ids) & trips["Destino_Id"].isin(valid_ids)].copy()

    hourly = _hourly_event_summary(trips)
    bounds = (
        pd.concat(
            [
                trips[["Origen_Id", "Inicio_del_viaje"]].rename(
                    columns={"Origen_Id": "station_id", "Inicio_del_viaje": "time"}
                ),
                trips[["Destino_Id", "Fin_del_viaje"]].rename(
                    columns={"Destino_Id": "station_id", "Fin_del_viaje": "time"}
                ),
            ],
            ignore_index=True,
        )
        .groupby("station_id")["time"]
        .agg(["min", "max"])
    )

    panels = []
    for station_id, row in bounds.iterrows():
        hours = pd.date_range(row["min"].floor("D"), row["max"].ceil("D"), freq="h")
        panels.append(pd.DataFrame({"station_id": station_id, "hour": hours}))
    panel = pd.concat(panels, ignore_index=True)
    panel = panel.merge(hourly, on=["station_id", "hour"], how="left")
    count_columns = [
        "departures",
        "arrivals",
        "max_bike_imbalance",
        "min_bike_imbalance",
    ]
    panel[count_columns] = panel[count_columns].fillna(0).astype("int16")
    panel = panel.merge(station_table, on="station_id", how="inner")
    panel["net_departures"] = panel["departures"] - panel["arrivals"]
    panel["risk_no_bikes"] = (
        panel["max_bike_imbalance"].clip(lower=0) >= alpha * panel["capacity"]
    ).astype("int8")
    panel["risk_no_docks"] = (
        (-panel["min_bike_imbalance"].clip(upper=0)) >= alpha * panel["capacity"]
    ).astype("int8")

    grouped = panel.groupby("station_id", sort=False)
    for column in ["departures", "arrivals", "net_departures"]:
        panel[f"{column}_lag_1h"] = grouped[column].shift(1)
        for window in [3, 24, 168]:
            panel[f"{column}_mean_{window}h"] = grouped[column].transform(
                lambda values, size=window: values.shift(1).rolling(size, min_periods=1).mean()
            )

    previous_risk = grouped[["risk_no_bikes", "risk_no_docks"]].shift(1)
    panel["risk_no_bikes_lag_1h"] = previous_risk["risk_no_bikes"]
    panel["risk_no_docks_lag_1h"] = previous_risk["risk_no_docks"]
    panel["hour_sin"] = np.sin(2 * np.pi * panel["hour"].dt.hour / 24)
    panel["hour_cos"] = np.cos(2 * np.pi * panel["hour"].dt.hour / 24)
    panel["weekday_sin"] = np.sin(2 * np.pi * panel["hour"].dt.dayofweek / 7)
    panel["weekday_cos"] = np.cos(2 * np.pi * panel["hour"].dt.dayofweek / 7)
    panel["month_sin"] = np.sin(2 * np.pi * (panel["hour"].dt.month - 1) / 12)
    panel["month_cos"] = np.cos(2 * np.pi * (panel["hour"].dt.month - 1) / 12)
    panel["is_weekend"] = panel["hour"].dt.dayofweek.ge(5).astype("int8")

    # The current-hour counts and imbalances define the labels and must never be features.
    return panel.dropna().sort_values(["hour", "station_id"]).reset_index(drop=True)


def build_hourly_features(trips: pd.DataFrame) -> pd.DataFrame:
    """Build station-hour demand targets and leakage-safe time features."""
    required_columns = {TIMESTAMP_COL, STATION_COL}
    missing = required_columns - set(trips.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = trips[[TIMESTAMP_COL, STATION_COL]].copy()
    data[TIMESTAMP_COL] = pd.to_datetime(data[TIMESTAMP_COL], errors="coerce")
    data[STATION_COL] = pd.to_numeric(data[STATION_COL], errors="coerce")

    data = data.dropna(subset=[TIMESTAMP_COL, STATION_COL])
    data[STATION_COL] = data[STATION_COL].astype(int)

    # Each trip contributes one departure to its station-hour.
    data["timestamp"] = data[TIMESTAMP_COL].dt.floor("h")

    hourly = (
        data.groupby([STATION_COL, "timestamp"])
        .size()
        .rename("target_departures")
        .reset_index()
        .rename(columns={STATION_COL: "station_id"})
    )

    # Fill zero-departure hours only within each station's observed interval.
    station_bounds = hourly.groupby("station_id")["timestamp"].agg(["min", "max"])
    station_panels = []

    for station_id, bounds in station_bounds.iterrows():
        timestamps = pd.date_range(
            start=bounds["min"],
            end=bounds["max"],
            freq="h",
        )
        station_panels.append(pd.DataFrame({"station_id": station_id, "timestamp": timestamps}))

    complete_index = (
        pd.concat(station_panels, ignore_index=True).set_index(["station_id", "timestamp"]).index
    )

    hourly = (
        hourly.set_index(["station_id", "timestamp"])
        .reindex(complete_index, fill_value=0)
        .rename_axis(["station_id", "timestamp"])
        .reset_index()
        .sort_values(["station_id", "timestamp"])
        .reset_index(drop=True)
    )

    hourly["hour"] = hourly["timestamp"].dt.hour
    hourly["day_of_week"] = hourly["timestamp"].dt.dayofweek
    hourly["month"] = hourly["timestamp"].dt.month
    hourly["is_weekend"] = (hourly["day_of_week"] >= 5).astype("int8")

    # Lag and rolling features use only previously completed hours.
    grouped = hourly.groupby("station_id")["target_departures"]

    hourly["departures_lag_1"] = grouped.shift(1)
    hourly["departures_lag_24"] = grouped.shift(24)
    hourly["departures_lag_168"] = grouped.shift(168)

    hourly["departures_mean_24"] = grouped.transform(
        lambda series: series.shift(1).rolling(window=24, min_periods=24).mean()
    )

    hourly["departures_mean_168"] = grouped.transform(
        lambda series: series.shift(1).rolling(window=168, min_periods=168).mean()
    )

    # Initial rows lack sufficient history for all lag features.
    feature_columns = [
        "departures_lag_1",
        "departures_lag_24",
        "departures_lag_168",
        "departures_mean_24",
        "departures_mean_168",
    ]
    hourly = hourly.dropna(subset=feature_columns).reset_index(drop=True)

    return hourly


@app.command("build-p1")
def build_p1_command(
    input_path: Path = P1_TRIPS_FILE,
    output_path: Path = P1_FEATURES_FILE,
):
    """Generate leakage-safe station-hour regression features."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    logger.info("Reading trips from {}", input_path)
    trips = pd.read_parquet(input_path)

    logger.info("Building hourly station features")
    features = build_hourly_features(trips)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False, compression="snappy")

    logger.success("Features saved to {}", output_path)
    logger.info("Rows: {}", f"{len(features):,}")
    logger.info("Stations: {}", features["station_id"].nunique())
    logger.info(
        "Time range: {} to {}",
        features["timestamp"].min(),
        features["timestamp"].max(),
    )


def count_active_days(trips: pd.DataFrame) -> pd.Series:
    """Count days with a departure or arrival at each station."""
    day = trips["inicio"].dt.normalize()
    days = pd.concat(
        [
            pd.DataFrame({"estacion_id": trips["origen_id"], "dia": day}),
            pd.DataFrame({"estacion_id": trips["destino_id"], "dia": day}),
        ]
    )
    return days.drop_duplicates().groupby("estacion_id").size()


def build_station_profile(trips: pd.DataFrame) -> pd.DataFrame:
    """Build the P3 usage profile with one row per station."""
    origin = trips["origen_id"].rename("estacion_id")
    destination = trips["destino_id"].rename("estacion_id")
    profile = (
        pd.DataFrame(
            {
                "dias_activos": count_active_days(trips),
                "salidas": trips.groupby(origin).size(),
                "llegadas": trips.groupby(destination).size(),
            }
        )
        .fillna(0)
        .astype(int)
    )
    total = profile["salidas"] + profile["llegadas"]
    profile["viajes_dia"] = total / profile["dias_activos"]
    profile["balance"] = (profile["llegadas"] - profile["salidas"]) / total

    time_range = trips["inicio"].dt.hour.map(P3_HOUR_TO_RANGE)
    by_range = pd.crosstab(origin, time_range, normalize="index")
    profile = profile.join(by_range.reindex(columns=list(P3_TIME_RANGES)).add_prefix("prop_"))
    profile["prop_fin_semana"] = (trips["inicio"].dt.dayofweek >= 5).groupby(origin).mean()
    profile["prop_redondo"] = (trips["origen_id"] == trips["destino_id"]).groupby(origin).mean()
    profile["duracion_mediana"] = trips.groupby(origin)["duracion_min"].median()
    return profile.rename_axis("estacion_id").reset_index()


def filter_station_profile(
    profile: pd.DataFrame, stations: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter inactive P3 stations and stations with insufficient history."""
    summary = {"estaciones_nomenclatura": len(stations)}
    profile = stations.merge(profile, on="estacion_id", how="inner")
    summary["sin_viajes"] = summary["estaciones_nomenclatura"] - len(profile)
    summary["retiradas"] = int((~profile["activa"]).sum())
    profile = profile[profile["activa"]]
    insufficient = profile["dias_activos"] < CLUSTERING_MIN_ACTIVE_DAYS
    summary["pocos_dias_activos"] = int(insufficient.sum())
    profile = profile[~insufficient]
    summary["estaciones_finales"] = len(profile)
    return profile.drop(columns="activa").reset_index(drop=True), summary


@app.command("build-p3")
def build_p3_command(
    trips_path: Path = CLUSTERING_TRIPS_FILE,
    stations_path: Path = CLUSTERING_STATIONS_CLEAN_FILE,
    output_path: Path = CLUSTERING_FEATURES_FILE,
) -> None:
    """Generate the station-level clustering features."""
    for path in [trips_path, stations_path]:
        if not path.exists():
            raise FileNotFoundError(f"No existe {path}. Corre primero dataset prepare-p3.")
    trips = pd.read_parquet(trips_path)
    stations = pd.read_parquet(stations_path)
    profile, summary = filter_station_profile(build_station_profile(trips), stations)
    for reason, count in summary.items():
        logger.info(f"{reason}: {count}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_parquet(output_path, index=False)
    logger.success(f"Perfil de {len(profile)} estaciones en {output_path}")


if __name__ == "__main__":
    app()
