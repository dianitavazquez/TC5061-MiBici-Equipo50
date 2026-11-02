"""Pruebas unitarias del entrenamiento de los modelos P1, P2 y P3 de MiBici.

Se utilizan datos sintéticos pequeños; no se requieren archivos reales ni un servidor MLflow.
Las pruebas comprueban métricas, casos límite y reproducibilidad con semillas fijas.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from mibici_ml.config import CLUSTERING_SEED, RANDOM_SEED
from mibici_ml.modeling.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGETS,
    build_p3_model,
    build_pipeline,
    chronological_split,
    evaluate,
    evaluate_clustering,
    evaluate_model,
    summarize_clusters,
)


@pytest.fixture
def datos_p2():
    n = 40
    indices = np.arange(n)
    datos = pd.DataFrame({columna: (indices % 7).astype(float) for columna in NUMERIC_FEATURES})
    for columna in CATEGORICAL_FEATURES:
        datos[columna] = (indices % 4 + 1).astype(str)
    objetivos = pd.DataFrame(
        {TARGETS[0]: indices % 2, TARGETS[1]: (indices // 2) % 2}
    )
    return datos, objetivos


@pytest.fixture
def datos_p3():
    from mibici_ml.modeling.train import P3_USAGE_FEATURES

    rng = np.random.default_rng(12)
    columnas = list(dict.fromkeys([*P3_USAGE_FEATURES, "capacidad"]))
    grupos = []
    for centro in (2.0, 12.0, 30.0):
        bloque = pd.DataFrame(
            rng.uniform(0.0, 0.3, size=(12, len(columnas))), columns=columnas
        )
        for columna in columnas:
            bloque[columna] += centro
        grupos.append(bloque)
    return pd.concat(grupos, ignore_index=True)


# P1: regresión y evaluación de demanda

def test_p1_evaluacion_calcula_mae_y_rmse():
    modelo = DummyRegressor(strategy="constant", constant=2.0)
    entrenamiento = pd.DataFrame({"x": [0, 1, 2]})
    modelo.fit(entrenamiento, [1, 2, 3])

    metricas, predicciones = evaluate_model(modelo, entrenamiento, np.array([1, 2, 3]))

    np.testing.assert_allclose(predicciones, [2, 2, 2])
    assert metricas["mae"] == pytest.approx(2 / 3)
    assert metricas["rmse"] == pytest.approx(np.sqrt(2 / 3))


def test_p1_evaluacion_no_permite_demanda_negativa():
    modelo = DummyRegressor(strategy="constant", constant=-5.0)
    datos = pd.DataFrame({"x": [0, 1, 2]})
    modelo.fit(datos, [0, 0, 0])

    metricas, predicciones = evaluate_model(modelo, datos, np.zeros(3))

    np.testing.assert_array_equal(predicciones, np.zeros(3))
    assert metricas == {"mae": 0.0, "rmse": 0.0}


def test_p1_entrenamiento_es_reproducible_con_semilla_fija():
    rng = np.random.default_rng(42)
    datos = rng.normal(size=(90, 4))
    objetivos = np.maximum(0, datos[:, 0] * 3 + datos[:, 1] + 5)
    parametros = dict(max_iter=20, min_samples_leaf=5, early_stopping=False, random_state=RANDOM_SEED)

    primero = HistGradientBoostingRegressor(**parametros).fit(datos, objetivos)
    segundo = HistGradientBoostingRegressor(**parametros).fit(datos, objetivos)

    np.testing.assert_array_equal(primero.predict(datos), segundo.predict(datos))


# P2: clasificación multisalida y partición cronológica

def test_p2_pipeline_entrena_y_predice_ambos_objetivos(datos_p2):
    datos, objetivos = datos_p2
    modelo = build_pipeline(max_iter=100).fit(datos, objetivos)

    predicciones = modelo.predict(datos)

    assert predicciones.shape == (len(datos), len(TARGETS))
    assert set(np.unique(predicciones)).issubset({0, 1})


def test_p2_entrenamiento_es_reproducible(datos_p2):
    datos, objetivos = datos_p2
    primero = build_pipeline(max_iter=100).fit(datos, objetivos)
    segundo = build_pipeline(max_iter=100).fit(datos, objetivos)

    np.testing.assert_array_equal(primero.predict(datos), segundo.predict(datos))
    for probabilidades_a, probabilidades_b in zip(
        primero.predict_proba(datos), segundo.predict_proba(datos)
    ):
        np.testing.assert_allclose(probabilidades_a, probabilidades_b)


def test_p2_metricas_perfectas_para_ambos_objetivos():
    objetivos = pd.DataFrame({TARGETS[0]: [0, 1, 0, 1], TARGETS[1]: [1, 0, 1, 0]})
    predicciones = objetivos.to_numpy()
    probabilidades = [
        np.column_stack((1 - objetivos[columna], objetivos[columna])).astype(float)
        for columna in TARGETS
    ]

    metricas = evaluate(objetivos, predicciones, probabilidades)

    assert metricas["macro_f1"] == pytest.approx(1.0)
    for columna in TARGETS:
        assert metricas[f"{columna}_roc_auc"] == pytest.approx(1.0)
        assert metricas[f"{columna}_precision"] == pytest.approx(1.0)


@pytest.mark.parametrize("fraccion", [0, 1, -0.2, 1.2])
def test_p2_particion_rechaza_fracciones_invalidas(fraccion):
    datos = pd.DataFrame({"hour": pd.date_range("2026-01-01", periods=5, freq="h")})
    with pytest.raises(ValueError, match="train_fraction"):
        chronological_split(datos, train_fraction=fraccion)


def test_p2_particion_rechaza_un_solo_instante():
    datos = pd.DataFrame({"hour": [pd.Timestamp("2026-01-01")]})
    with pytest.raises(ValueError, match="conjunto vacío"):
        chronological_split(datos, train_fraction=0.8)


# P3: clustering, reproducibilidad y métricas

def test_p3_entrenamiento_asigna_un_grupo_a_cada_estacion(datos_p3):
    from mibici_ml.modeling.train import P3_USAGE_FEATURES

    datos = datos_p3[P3_USAGE_FEATURES]
    modelo = build_p3_model(k=3, n_init=5).fit(datos)
    grupos = modelo.predict(datos)

    assert len(grupos) == len(datos)
    assert len(np.unique(grupos)) == 3
    assert modelo.named_steps["kmeans"].random_state == CLUSTERING_SEED


def test_p3_entrenamiento_es_reproducible(datos_p3):
    from mibici_ml.modeling.train import P3_USAGE_FEATURES

    datos = datos_p3[P3_USAGE_FEATURES]
    primero = build_p3_model(k=3, n_init=5).fit(datos)
    segundo = build_p3_model(k=3, n_init=5).fit(datos)

    np.testing.assert_array_equal(primero.predict(datos), segundo.predict(datos))
    np.testing.assert_allclose(
        primero.named_steps["kmeans"].cluster_centers_,
        segundo.named_steps["kmeans"].cluster_centers_,
    )


def test_p3_metricas_de_agrupamiento_son_validas():
    datos = np.array([[0, 0], [0, 1], [10, 10], [10, 11], [20, 20], [20, 21]])
    grupos = np.array([0, 0, 1, 1, 2, 2])

    metricas = evaluate_clustering(datos, grupos)

    assert -1 <= metricas["silhouette"] <= 1
    assert metricas["davies_bouldin"] >= 0
    assert metricas["calinski_harabasz"] > 0
    assert metricas["grupo_mas_chico_pct"] == pytest.approx(100 / 3)


def test_p3_resumen_conserva_total_de_estaciones(datos_p3):
    from mibici_ml.modeling.train import P3_USAGE_FEATURES

    grupos = np.repeat([0, 1, 2], 12)
    resumen = summarize_clusters(datos_p3, grupos)

    assert resumen["n_estaciones"].sum() == len(datos_p3)
    assert resumen["n_estaciones"].tolist() == [12, 12, 12]
    assert set(P3_USAGE_FEATURES).issubset(resumen.columns)
