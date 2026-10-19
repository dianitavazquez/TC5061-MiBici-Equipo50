import pandas as pd

from mibici_ml.features import build_station_hour_dataset
from mibici_ml.modeling.train import chronological_split


def test_targets_use_ordered_intra_hour_imbalance():
    trips = pd.DataFrame(
        {
            "Inicio_del_viaje": pd.to_datetime(
                ["2026-01-01 08:05", "2026-01-01 08:10", "2026-01-01 08:15"]
            ),
            "Fin_del_viaje": pd.to_datetime(
                ["2026-01-01 09:05", "2026-01-01 09:10", "2026-01-01 09:15"]
            ),
            "Origen_Id": [1, 1, 1],
            "Destino_Id": [2, 2, 2],
        }
    )
    stations = pd.DataFrame({"id": [1, 2], "dpcapacity": [4, 4]})

    result = build_station_hour_dataset(trips, stations, alpha=0.5)

    origin = result[(result["station_id"] == 1) & (result["hour"].dt.hour == 8)].iloc[0]
    destination = result[(result["station_id"] == 2) & (result["hour"].dt.hour == 9)].iloc[0]
    assert origin["risk_no_bikes"] == 1
    assert origin["risk_no_docks"] == 0
    assert destination["risk_no_docks"] == 1


def test_chronological_split_keeps_future_in_test():
    hours = pd.date_range("2026-01-01", periods=10, freq="h")
    frame = pd.DataFrame({"hour": hours, "value": range(10)})

    train, test, cutoff = chronological_split(frame, train_fraction=0.8)

    assert train["hour"].max() < cutoff
    assert test["hour"].min() == cutoff
    assert train["hour"].max() < test["hour"].min()
