"""Pruebas de la validación de datos de P3 con DeepChecks (validation.py).

Las primeras arman tablas pequeñas con problemas puestos a propósito para comprobar que las
suites sí detectan nulos, duplicados, tipos mezclados y drift. Las dos últimas revisan los datos
reales de P3 y se saltan si todavía no se corre la ingesta.
"""

import json

import numpy as np
import pandas as pd
import pytest

from mibici_ml.config import (
    CLUSTERING_FEATURES_FILE,
    CLUSTERING_MAX_DUPLICATE_RATIO,
    CLUSTERING_TRIPS_FILE,
)
from mibici_ml.validation import (
    drift_datasets,
    drift_suite,
    integrity_suite,
    profile_by_period,
    profile_dataset,
    split_by_period,
    summarize,
    trips_dataset,
)

requiere_datos_reales = pytest.mark.skipif(
    not (CLUSTERING_TRIPS_FILE.exists() and CLUSTERING_FEATURES_FILE.exists()),
    reason="Faltan los datos de P3; corre dataset prepare-p3 y features build-p3",
)


def viajes_limpios(n=300, semilla=0):
    """Arma viajes al azar con las columnas y tipos que deja clean_p3_trips."""
    rng = np.random.default_rng(semilla)
    segundos = rng.integers(0, 30 * 24 * 3600, n)
    inicio = pd.Timestamp("2026-03-01") + pd.to_timedelta(segundos, unit="s")
    duracion = rng.integers(2, 40, n)
    return pd.DataFrame(
        {
            "inicio": inicio,
            "fin": inicio + pd.to_timedelta(duracion, unit="min"),
            "origen_id": rng.integers(1, 6, n).astype("int32"),
            "destino_id": rng.integers(1, 6, n).astype("int32"),
            "duracion_min": duracion.astype(float),
        }
    )


def tabla_para_drift(media, n=400, semilla=0):
    """Arma una variable numérica con la media indicada y otra que no cambia."""
    rng = np.random.default_rng(semilla)
    return pd.DataFrame({"duracion_min": rng.normal(media, 3, n), "hora": rng.integers(0, 24, n)})


def chequeos_fallidos(resumen):
    return {nombre for nombre, chequeo in resumen["chequeos"].items() if not chequeo["aprobado"]}


def revisar_integridad(viajes, **parametros):
    return summarize(integrity_suite(**parametros).run(trips_dataset(viajes)))


# ---------------------------------------------------------------------------
# Integridad: tipos, nulos y duplicados
# ---------------------------------------------------------------------------


def test_integridad_aprueba_viajes_sin_problemas():
    resumen = revisar_integridad(viajes_limpios())

    assert resumen["aprobado"]
    assert chequeos_fallidos(resumen) == set()
    json.dumps(resumen)  # el resumen se debe poder guardar como JSON


def test_integridad_detecta_nulos():
    viajes = viajes_limpios()
    viajes["destino_id"] = viajes["destino_id"].where(viajes.index >= 15)

    resumen = revisar_integridad(viajes)

    assert chequeos_fallidos(resumen) == {"PercentOfNulls"}
    assert resumen["chequeos"]["PercentOfNulls"]["valor"]["destino_id"] == pytest.approx(0.05)


def test_integridad_detecta_nulos_escritos_de_distintas_formas():
    viajes = viajes_limpios().astype({"destino_id": object})
    viajes.loc[:4, "destino_id"] = np.nan
    viajes.loc[5:9, "destino_id"] = "NULL"

    assert "Mixed Nulls" in chequeos_fallidos(revisar_integridad(viajes))


def test_integridad_detecta_tipos_mezclados():
    viajes = viajes_limpios().astype({"duracion_min": object})
    viajes.loc[:9, "duracion_min"] = "sin dato"

    assert chequeos_fallidos(revisar_integridad(viajes)) == {"Mixed Data Types"}


def test_integridad_detecta_filas_duplicadas():
    viajes = viajes_limpios()
    viajes = pd.concat([viajes, viajes.head(6)], ignore_index=True)

    resumen = revisar_integridad(viajes)

    assert chequeos_fallidos(resumen) == {"Data Duplicates"}
    assert resumen["chequeos"]["Data Duplicates"]["valor"] == pytest.approx(6 / 306, abs=1e-6)


def test_integridad_tolera_duplicados_por_debajo_del_limite():
    viajes = viajes_limpios()
    viajes = pd.concat([viajes, viajes.head(1)], ignore_index=True)

    assert not revisar_integridad(viajes)["aprobado"]
    assert revisar_integridad(viajes, max_duplicate_ratio=0.01)["aprobado"]


def test_integridad_del_perfil_no_revisa_el_nombre_de_la_estacion():
    perfil = pd.DataFrame(
        {
            "estacion_id": [1, 2, 3, 4],
            "nombre": ["(GDL-001) A", "(GDL-002) B", "(ZPN-003) C", "(TLQ-004) D"],
            "zona": ["GDL", "GDL", "ZPN", "TLQ"],
            "viajes_dia": [40.5, 12.0, 8.3, 25.1],
        }
    )

    dataset = profile_dataset(perfil)

    assert "nombre" not in dataset.data.columns
    assert dataset.index_name == "estacion_id"


# ---------------------------------------------------------------------------
# Drift entre train y test
# ---------------------------------------------------------------------------


def test_drift_aprueba_dos_periodos_con_la_misma_distribucion():
    train, test = drift_datasets(
        tabla_para_drift(10, semilla=1), tabla_para_drift(10, semilla=2), []
    )

    resumen = summarize(drift_suite().run(train, test))

    assert resumen["aprobado"]


def test_drift_detecta_un_cambio_en_una_variable():
    test = tabla_para_drift(10, semilla=2)
    test["duracion_min"] += 5
    train, test = drift_datasets(tabla_para_drift(10, semilla=1), test, [])

    resumen = summarize(drift_suite().run(train, test))

    assert chequeos_fallidos(resumen) == {"Feature Drift", "Multivariate Drift"}
    puntajes = resumen["chequeos"]["Feature Drift"]["valor"]
    assert puntajes["duracion_min"] > 0.2 > puntajes["hora"]


def test_separar_por_periodo_respeta_la_fecha_de_corte():
    viajes = pd.DataFrame(
        {
            "inicio": pd.to_datetime(
                ["2026-06-30 23:59:59", "2026-07-01 00:00:00", "2026-07-15 08:00:00"]
            )
        }
    )

    train, test = split_by_period(viajes, "2026-07-01")

    assert train["inicio"].tolist() == [pd.Timestamp("2026-06-30 23:59:59")]
    assert len(test) == 2


def test_perfil_por_periodo_solo_compara_estaciones_presentes_en_ambos():
    filas = []
    for dia in pd.date_range("2026-06-01 08:00", "2026-07-31 08:00", freq="D"):
        filas += [(dia, 1, 2), (dia, 2, 1)]
        if dia >= pd.Timestamp("2026-07-01"):
            filas.append((dia, 3, 1))  # estación que abrió en julio
    viajes = pd.DataFrame(filas, columns=["inicio", "origen_id", "destino_id"])
    viajes["fin"] = viajes["inicio"] + pd.Timedelta(minutes=10)
    viajes["duracion_min"] = 10.0
    estaciones = pd.DataFrame({"estacion_id": [1, 2, 3], "activa": [True, True, True]})

    train, test = profile_by_period(viajes, estaciones, "2026-07-01")

    assert set(train["estacion_id"]) == set(test["estacion_id"]) == {1, 2}


# ---------------------------------------------------------------------------
# Datos reales de P3
# ---------------------------------------------------------------------------


@requiere_datos_reales
def test_perfil_real_pasa_la_integridad():
    perfil = pd.read_parquet(CLUSTERING_FEATURES_FILE)

    assert summarize(integrity_suite().run(profile_dataset(perfil)))["aprobado"]


@requiere_datos_reales
def test_viajes_limpios_reales_no_tienen_nulos_y_casi_no_tienen_duplicados():
    viajes = pd.read_parquet(CLUSTERING_TRIPS_FILE)

    resumen = revisar_integridad(viajes, max_duplicate_ratio=CLUSTERING_MAX_DUPLICATE_RATIO)

    assert resumen["aprobado"]
    assert set(resumen["chequeos"]["PercentOfNulls"]["valor"].values()) == {0.0}
