import numpy as np
import pandas as pd

from mibici_ml.features import build_station_hour_dataset
from mibici_ml.modeling.train import chronological_split, evaluate_monthly_stability


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


def test_monthly_stability_reports_same_p2_targets():
    test = pd.DataFrame(
        {
            "hour": pd.to_datetime(["2026-07-01", "2026-07-02", "2026-08-01", "2026-08-02"]),
            "risk_no_bikes": [0, 1, 0, 1],
            "risk_no_docks": [1, 0, 1, 0],
        }
    )
    predictions = np.array([[0, 1], [1, 0], [0, 1], [0, 0]])

    metrics, monthly = evaluate_monthly_stability(test, predictions)

    assert monthly["month"].tolist() == ["2026-07", "2026-08"]
    assert monthly["macro_f1"].tolist() == [1.0, 0.5]
    assert metrics["monthly_macro_f1_mean"] == 0.75
    assert metrics["monthly_macro_f1_std"] == 0.25
