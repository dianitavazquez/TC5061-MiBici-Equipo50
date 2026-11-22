"""Unit tests for the P2 champion-versus-challenger A/B test."""

import json

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from mibici_ml.modeling import ab_testing
from mibici_ml.modeling.train import NUMERIC_FEATURES, TARGETS, build_pipeline


def periodo_actual(horas=72, estaciones=6, semilla=0):
    """Panel estación-hora sintético del 15 de octubre en adelante."""
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2026-10-15", periods=horas, freq="h")
    frame = pd.DataFrame(
        [(estacion, fecha) for estacion in range(1, estaciones + 1) for fecha in fechas],
        columns=["station_id", "hour"],
    )
    for columna in NUMERIC_FEATURES:
        frame[columna] = rng.normal(size=len(frame))
    frame["capacity"] = rng.integers(10, 30, size=len(frame))
    frame["risk_no_bikes"] = (frame["departures_lag_1h"] > 1).astype("int8")
    frame["risk_no_docks"] = (frame["arrivals_lag_1h"] > 1).astype("int8")
    return frame


def test_macro_f1_por_conteos_coincide_con_sklearn():
    rng = np.random.default_rng(1)
    real = rng.integers(0, 2, size=(200, 2))
    prediccion = rng.integers(0, 2, size=(200, 2))
    conteos = ab_testing.station_confusion_counts(rng.integers(0, 7, size=200), real, prediccion)
    esperado = np.mean([f1_score(real[:, i], prediccion[:, i]) for i in range(2)])

    assert conteos.shape == (7, 2, 3)
    assert ab_testing.macro_f1_from_counts(conteos.sum(axis=0)) == pytest.approx(esperado)


def test_macro_f1_es_cero_sin_positivos_ni_alarmas():
    assert ab_testing.macro_f1_from_counts(np.zeros((2, 3))) == 0


def test_bootstrap_de_modelos_identicos_da_diferencia_cero():
    frame = periodo_actual()
    prediccion = frame[TARGETS].to_numpy()

    resumen, diferencias = ab_testing.paired_station_bootstrap(
        frame, prediccion, prediccion, samples=50
    )

    assert np.all(diferencias == 0)
    assert resumen["ci_low"] == resumen["ci_high"] == 0
    assert resumen["probability_challenger_better"] == 0
    assert resumen["stations"] == 6


def test_bootstrap_es_reproducible_y_detecta_una_mejora_clara():
    frame = periodo_actual()
    real = frame[TARGETS].to_numpy()
    azar = np.random.default_rng(3).integers(0, 2, size=real.shape)

    primero, _ = ab_testing.paired_station_bootstrap(frame, azar, real, samples=100, seed=7)
    segundo, _ = ab_testing.paired_station_bootstrap(frame, azar, real, samples=100, seed=7)

    assert primero == segundo
    assert primero["ci_low"] > 0


def recall(sin_bicis, sin_anclajes):
    return {"risk_no_bikes_recall": sin_bicis, "risk_no_docks_recall": sin_anclajes}


@pytest.mark.parametrize(
    "intervalo, recall_challenger, ganador, veredicto",
    [
        ((0.01, 0.05), 0.28, "challenger", "es mejor"),
        ((-0.01, 0.05), 0.30, "champion", "no es concluyente"),
        ((-0.09, -0.01), 0.30, "champion", "es peor"),
        ((0.01, 0.05), 0.20, "champion", "es mejor"),
    ],
)
def test_decision_exige_mejora_clara_sin_perder_recall(
    intervalo, recall_challenger, ganador, veredicto
):
    bootstrap = {"ci_low": intervalo[0], "ci_high": intervalo[1], "confidence": 0.95}

    decision = ab_testing.decide(
        recall(0.30, 0.30), recall(recall_challenger, 0.30), bootstrap, max_recall_drop=0.05
    )

    assert decision["winner"] == ganador
    assert decision["promote_challenger"] is (ganador == "challenger")
    assert veredicto in decision["reasons"][0]


def test_ventana_de_evaluacion_filtra_fechas(tmp_path):
    ruta = tmp_path / "actual.parquet"
    periodo_actual().to_parquet(ruta)

    frame = ab_testing.load_evaluation_frame(ruta, "2026-10-16", "2026-11-01")

    assert frame["hour"].min() == pd.Timestamp("2026-10-16")
    assert len(frame) == 6 * 48


def test_ventana_invertida_vacia_o_sin_archivo_falla(tmp_path):
    ruta = tmp_path / "actual.parquet"
    periodo_actual().to_parquet(ruta)

    with pytest.raises(ValueError, match="anterior"):
        ab_testing.load_evaluation_frame(ruta, "2026-11-01", "2026-10-16")
    with pytest.raises(ValueError, match="No hay filas"):
        ab_testing.load_evaluation_frame(ruta, "2026-12-01", "2026-12-31")
    with pytest.raises(FileNotFoundError):
        ab_testing.load_evaluation_frame(tmp_path / "no_existe.parquet")


def test_origen_del_modelo_lee_metadatos_del_registry(tmp_path):
    (tmp_path / "model-info.json").write_text(
        json.dumps({"registered_model": "p2", "version": 3, "run_id": "abc"}), encoding="utf-8"
    )

    assert ab_testing.describe_model_source(str(tmp_path)) == {
        "uri": tmp_path.name,
        "registered_model": "p2",
        "version": 3,
        "run_id": "abc",
    }
    assert ab_testing.describe_model_source("models:/p2/3") == {"uri": "models:/p2/3"}


def test_ab_completo_escribe_tabla_figura_y_resumen(tmp_path):
    frame = periodo_actual(horas=96)
    ruta = tmp_path / "actual.parquet"
    frame.to_parquet(ruta)
    entrenamiento = frame[frame["hour"] < pd.Timestamp("2026-10-16")]
    modelos = {}
    for nombre, c in [("champion", 0.01), ("challenger", 1.0)]:
        modelo = build_pipeline(c=c).fit(
            entrenamiento[ab_testing.FEATURES], entrenamiento[TARGETS]
        )
        modelos[nombre] = tmp_path / nombre
        mlflow.sklearn.save_model(
            modelo,
            str(modelos[nombre]),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    resumen = ab_testing.run_ab_test(
        champion_uri=str(modelos["champion"]),
        challenger_uri=str(modelos["challenger"]),
        evaluation_path=ruta,
        start="2026-10-16",
        samples=50,
        summary_dir=tmp_path / "resumen",
        figure_dir=tmp_path / "figuras",
        output_name="prueba",
        log_to_mlflow=False,
    )

    guardado = json.loads((tmp_path / "resumen" / "prueba.json").read_text(encoding="utf-8"))
    tabla = pd.read_csv(tmp_path / "resumen" / "prueba.csv", index_col="metric")
    assert guardado == json.loads(json.dumps(resumen))
    assert guardado["evaluation"]["rows"] == 6 * 72
    assert guardado["decision"]["winner"] in ab_testing.VARIANTS
    assert "mlflow_run_id" not in guardado
    assert tabla.loc["macro_f1", "difference"] == pytest.approx(
        tabla.loc["macro_f1", "challenger"] - tabla.loc["macro_f1", "champion"], abs=1e-5
    )
    assert (tmp_path / "figuras" / "prueba.png").stat().st_size > 0
