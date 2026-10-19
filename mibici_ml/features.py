"""Feature and target construction for the station-hour P2 classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

from mibici_ml.config import PROCESSED_DATA_DIR
from mibici_ml.dataset import TRIP_COLUMNS, read_csv_compatible

app = typer.Typer(no_args_is_help=True)

TIMESTAMP_COL = "Inicio_del_viaje"
STATION_COL = "Origen_Id"


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
    input_path: Path = PROCESSED_DATA_DIR / "mibici_trips.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "p1_features.parquet",
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


if __name__ == "__main__":
    app()
