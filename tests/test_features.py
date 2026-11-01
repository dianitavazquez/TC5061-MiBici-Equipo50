from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mibici_ml import features
from mibici_ml.features import (
    _hourly_event_summary,
    build_hourly_features,
    build_station_hour_dataset,
    build_station_profile,
    count_active_days,
    filter_station_profile,
    load_clean_trips,
)


def _p2_trips(**overrides) -> pd.DataFrame:
    data = {
        "Inicio_del_viaje": pd.to_datetime(
            ["2026-01-05 08:05", "2026-01-05 08:10", "2026-01-05 10:00"]
        ),
        "Fin_del_viaje": pd.to_datetime(
            ["2026-01-05 09:05", "2026-01-05 09:10", "2026-01-05 10:30"]
        ),
        "Origen_Id": [1, 1, 2],
        "Destino_Id": [2, 2, 1],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def _p3_trips() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "inicio": pd.to_datetime(
                [
                    "2026-01-05 07:00",
                    "2026-01-05 12:00",
                    "2026-01-10 17:00",
                    "2026-01-11 22:00",
                ]
            ),
            "fin": pd.to_datetime(
                [
                    "2026-01-05 07:10",
                    "2026-01-05 12:20",
                    "2026-01-10 17:30",
                    "2026-01-11 22:40",
                ]
            ),
            "origen_id": [1, 1, 1, 1],
            "destino_id": [2, 1, 2, 2],
            "duracion_min": [10.0, 20.0, 30.0, 40.0],
        }
    )


class TestTripLoading:
    def test_load_clean_trips_combines_files_and_removes_invalid_rows(self, tmp_path, monkeypatch):
        file_names = ["month_1.csv", "month_2.csv"]
        for name in file_names:
            (tmp_path / name).touch()

        frames = {
            "month_1.csv": pd.DataFrame(
                {
                    "Inicio_del_viaje": ["2026-01-01 08:00", "bad-date"],
                    "Fin_del_viaje": ["2026-01-01 08:15", "2026-01-01 09:00"],
                    "Origen_Id": [1, 2],
                    "Destino_Id": [2, 3],
                }
            ),
            "month_2.csv": pd.DataFrame(
                {
                    "Inicio_del_viaje": ["2026-02-01 10:00", "2026-02-01 11:00"],
                    "Fin_del_viaje": ["2026-02-01 09:59", "2026-02-01 11:10"],
                    "Origen_Id": [3, 4],
                    "Destino_Id": [5, 4],
                }
            ),
        }
        monkeypatch.setattr(features, "CLUSTERING_TRIP_FILES", file_names)
        monkeypatch.setattr(
            features,
            "read_csv_compatible",
            lambda path, usecols: frames[Path(path).name][usecols].copy(),
        )

        result, summary = load_clean_trips(tmp_path)

        assert summary == {"raw_trips": 4, "clean_trips": 1}
        assert result[["Origen_Id", "Destino_Id"]].to_dict("records") == [
            {"Origen_Id": 1, "Destino_Id": 2}
        ]
        assert result["Inicio_del_viaje"].dtype == "datetime64[ns]"
        assert result["Origen_Id"].dtype == "int32"

    def test_load_clean_trips_raises_when_a_source_file_is_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(features, "CLUSTERING_TRIP_FILES", ["missing.csv"])

        with pytest.raises(FileNotFoundError, match="missing.csv"):
            load_clean_trips(tmp_path)


class TestP1Features:
    def test_build_hourly_features_creates_calendar_and_historical_features(self):
        timestamps = pd.date_range("2026-01-01", periods=170, freq="h")
        trips = pd.DataFrame(
            {
                "Inicio_del_viaje": [*timestamps, timestamps[-1]],
                "Origen_Id": [1] * 171,
                "unused_personal_attribute": ["ignored"] * 171,
            }
        )

        result = build_hourly_features(trips)

        assert result["timestamp"].tolist() == [timestamps[168], timestamps[169]]
        assert result["target_departures"].tolist() == [1, 2]
        assert result["departures_lag_1"].tolist() == [1.0, 1.0]
        assert result["departures_lag_168"].tolist() == [1.0, 1.0]
        assert result["departures_mean_168"].tolist() == [1.0, 1.0]
        assert result.loc[0, "hour"] == timestamps[168].hour
        assert result.loc[0, "day_of_week"] == timestamps[168].dayofweek
        assert "unused_personal_attribute" not in result

    def test_build_hourly_features_fills_unobserved_hours_with_zero(self):
        timestamps = pd.date_range("2026-01-01", periods=170, freq="h")
        trips = pd.DataFrame(
            {
                "Inicio_del_viaje": timestamps.delete(10),
                "Origen_Id": 7,
            }
        )

        result = build_hourly_features(trips)

        first_row = result.iloc[0]
        assert first_row["timestamp"] == timestamps[168]
        assert first_row["departures_mean_168"] == pytest.approx(167 / 168)

    @pytest.mark.parametrize("missing_column", ["Inicio_del_viaje", "Origen_Id"])
    def test_build_hourly_features_rejects_missing_required_columns(self, missing_column):
        trips = pd.DataFrame(
            {
                "Inicio_del_viaje": ["2026-01-01 08:00"],
                "Origen_Id": [1],
            }
        ).drop(columns=missing_column)

        with pytest.raises(ValueError, match=missing_column):
            build_hourly_features(trips)

    def test_build_hourly_features_discards_invalid_values(self):
        timestamps = pd.date_range("2026-01-01", periods=170, freq="h")
        valid = pd.DataFrame({"Inicio_del_viaje": timestamps, "Origen_Id": 1})
        invalid = pd.DataFrame(
            {
                "Inicio_del_viaje": ["not-a-date", timestamps[0]],
                "Origen_Id": [1, "not-a-station"],
            }
        )

        result = build_hourly_features(pd.concat([valid, invalid], ignore_index=True))

        assert len(result) == 2
        assert result["station_id"].eq(1).all()


class TestP2Features:
    def test_hourly_summary_uses_ordered_imbalance_and_combines_simultaneous_events(self):
        trips = pd.DataFrame(
            {
                "Inicio_del_viaje": pd.to_datetime(
                    ["2026-01-01 08:05", "2026-01-01 08:05", "2026-01-01 08:20"]
                ),
                "Fin_del_viaje": pd.to_datetime(
                    ["2026-01-01 08:20", "2026-01-01 09:00", "2026-01-01 09:10"]
                ),
                "Origen_Id": [1, 1, 2],
                "Destino_Id": [2, 2, 1],
            }
        )

        result = _hourly_event_summary(trips)
        station_one = result[
            result["station_id"].eq(1) & result["hour"].eq(pd.Timestamp("2026-01-01 08:00"))
        ].iloc[0]

        assert station_one["departures"] == 2
        assert station_one["arrivals"] == 0
        assert station_one["max_bike_imbalance"] == 2
        assert station_one["min_bike_imbalance"] == 2

    def test_build_station_hour_dataset_creates_targets_lags_and_calendar_features(self):
        stations = pd.DataFrame({"id": [1, 2], "dpcapacity": [4, 4]})

        result = build_station_hour_dataset(_p2_trips(), stations, alpha=0.5)

        at_eight = result["hour"].eq(pd.Timestamp("2026-01-05 08:00"))
        at_nine = result["hour"].eq(pd.Timestamp("2026-01-05 09:00"))
        origin = result[result["station_id"].eq(1) & at_eight].iloc[0]
        next_hour = result[result["station_id"].eq(1) & at_nine].iloc[0]
        destination = result[result["station_id"].eq(2) & at_nine].iloc[0]
        assert origin["risk_no_bikes"] == 1
        assert destination["risk_no_docks"] == 1
        assert next_hour["departures_lag_1h"] == 2
        assert next_hour["risk_no_bikes_lag_1h"] == 1
        assert next_hour["departures_mean_3h"] == pytest.approx(2 / 3)
        assert origin["hour_sin"] == pytest.approx(np.sin(2 * np.pi * 8 / 24))
        assert origin["is_weekend"] == 0

    def test_build_station_hour_dataset_ignores_unknown_trips_and_invalid_stations(self):
        trips = _p2_trips()
        unknown_trip = pd.DataFrame(
            {
                "Inicio_del_viaje": pd.to_datetime(["2026-01-05 08:00"]),
                "Fin_del_viaje": pd.to_datetime(["2026-01-05 08:10"]),
                "Origen_Id": [99],
                "Destino_Id": [1],
            }
        )
        stations = pd.DataFrame({"id": [1, 2, 3, 4], "dpcapacity": [4, 4, 0, np.nan]})

        result = build_station_hour_dataset(
            pd.concat([trips, unknown_trip], ignore_index=True), stations
        )

        assert set(result["station_id"]) == {1, 2}
        assert result["departures"].sum() == 3

    @pytest.mark.parametrize("alpha", [0, -0.1, 1.01])
    def test_build_station_hour_dataset_rejects_alpha_outside_interval(self, alpha):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            build_station_hour_dataset(
                _p2_trips(), pd.DataFrame({"id": [1, 2], "dpcapacity": [4, 4]}), alpha
            )


class TestP3Features:
    def test_count_active_days_counts_unique_station_dates(self):
        result = count_active_days(_p3_trips())

        assert result.to_dict() == {1: 3, 2: 3}

    def test_build_station_profile_calculates_behavioral_features(self):
        result = build_station_profile(_p3_trips()).set_index("estacion_id")

        station = result.loc[1]
        assert station["dias_activos"] == 3
        assert station["salidas"] == 4
        assert station["llegadas"] == 1
        assert station["viajes_dia"] == pytest.approx(5 / 3)
        assert station["balance"] == pytest.approx(-3 / 5)
        assert station["prop_manana"] == 0.25
        assert station["prop_mediodia"] == 0.25
        assert station["prop_tarde"] == 0.25
        assert station["prop_noche"] == 0.25
        assert station["prop_fin_semana"] == 0.5
        assert station["prop_redondo"] == 0.25
        assert station["duracion_mediana"] == 25.0

    def test_build_station_profile_includes_destination_only_station(self):
        result = build_station_profile(_p3_trips()).set_index("estacion_id")

        station = result.loc[2]
        assert station["salidas"] == 0
        assert station["llegadas"] == 3
        assert station["balance"] == 1.0
        assert pd.isna(station["duracion_mediana"])

    def test_filter_station_profile_reports_each_exclusion_reason(self, monkeypatch):
        monkeypatch.setattr(features, "CLUSTERING_MIN_ACTIVE_DAYS", 3)
        profile = pd.DataFrame(
            {
                "estacion_id": [1, 2, 3],
                "dias_activos": [3, 5, 2],
                "viajes_dia": [1.0, 2.0, 3.0],
            }
        )
        stations = pd.DataFrame(
            {
                "estacion_id": [1, 2, 3, 4],
                "activa": [True, False, True, True],
                "nombre": ["A", "B", "C", "D"],
            }
        )

        result, summary = filter_station_profile(profile, stations)

        assert result["estacion_id"].tolist() == [1]
        assert "activa" not in result
        assert summary == {
            "estaciones_nomenclatura": 4,
            "sin_viajes": 1,
            "retiradas": 1,
            "pocos_dias_activos": 1,
            "estaciones_finales": 1,
        }
