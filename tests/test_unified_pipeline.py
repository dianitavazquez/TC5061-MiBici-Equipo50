import pandas as pd

from mibici_ml.features import build_station_profile, filter_station_profile
from mibici_ml.modeling.train import build_p3_model, evaluate_clustering


def test_p3_features_and_training_are_available_in_shared_modules():
    trips = pd.DataFrame(
        {
            "inicio": pd.to_datetime(["2026-01-05 07:00", "2026-01-05 12:00", "2026-01-06 17:00"]),
            "fin": pd.to_datetime(["2026-01-05 07:10", "2026-01-05 12:15", "2026-01-06 17:20"]),
            "origen_id": [1, 2, 1],
            "destino_id": [2, 1, 2],
            "duracion_min": [10.0, 15.0, 20.0],
        }
    )
    stations = pd.DataFrame(
        {
            "estacion_id": [1, 2],
            "nombre": ["GDL A", "GDL B"],
            "zona": ["GDL", "GDL"],
            "latitud": [20.67, 20.68],
            "longitud": [-103.34, -103.35],
            "capacidad": [10, 12],
            "activa": [True, True],
        }
    )

    profile = build_station_profile(trips)
    filtered, summary = filter_station_profile(profile, stations)

    assert set(profile["estacion_id"]) == {1, 2}
    assert filtered.empty  # ambas tienen menos de los 30 días mínimos configurados
    assert summary["pocos_dias_activos"] == 2
    assert build_p3_model(2).named_steps["kmeans"].n_clusters == 2


def test_clustering_metrics_use_shared_training_module():
    values = [[0.0, 0.0], [0.1, 0.2], [5.0, 5.0], [5.2, 5.1]]
    groups = [0, 0, 1, 1]

    metrics = evaluate_clustering(values, groups)

    assert metrics["silhouette"] > 0.9
    assert metrics["grupo_mas_chico_pct"] == 50.0
