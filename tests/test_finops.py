"""Unit tests for the FinOps cost model of the P2 API."""

import json

import numpy as np
import pandas as pd
import pytest

from mibici_ml import finops
from mibici_ml.config import MODELS_DIR

INICIO = 1_795_394_837


def prueba_de_carga(carpeta, escalones=((1, 200.0, 5, "88.00%"), (4, 120.0, 600, "99.00%"))):
    """Historia de Locust y docker stats sintéticos con escalones de 30 s."""
    historia = [
        {"Timestamp": INICIO - 1, "User Count": 0, "Name": "Aggregated", "Requests/s": 0.0}
    ]
    recursos = []
    for paso, (usuarios, rps, p95, cpu) in enumerate(escalones):
        for segundo in range(30):
            instante = INICIO + paso * 30 + segundo
            historia.append(
                {
                    "Timestamp": instante,
                    "User Count": usuarios,
                    "Name": "Aggregated",
                    "Requests/s": rps,
                    "Failures/s": 0.0,
                    "50%": p95 / 2,
                    "95%": p95,
                }
            )
            recursos.append(
                {
                    "timestamp": pd.Timestamp(instante, unit="s").strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "container": "finops-api",
                    "cpu_percent": cpu,
                    "memory_usage": "220MiB / 512MiB",
                    "memory_percent": "42.97%",
                }
            )
    carpeta.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(historia).to_csv(carpeta / "saturacion_stats_history.csv", index=False)
    pd.DataFrame(recursos).to_csv(carpeta / "recursos.csv", index=False)
    arranques = [{"hasta_health_s": valor} for valor in (4.7, 2.9, 2.8)]
    (carpeta / "arranque_frio.json").write_text(json.dumps(arranques), encoding="utf-8")
    return carpeta


def tabla_de_saturacion(carpeta):
    historia = finops.read_locust_history(carpeta / "saturacion_stats_history.csv")
    recursos = finops.read_docker_stats(carpeta / "recursos.csv")
    return finops.saturation_table(historia, recursos)


def test_memoria_de_docker_stats_se_convierte_a_mib():
    assert finops.parse_mib("210.6MiB") == pytest.approx(210.6)
    assert finops.parse_mib("2.114GiB") == pytest.approx(2164.736)
    assert finops.parse_mib("512KiB") == pytest.approx(0.5)
    with pytest.raises(ValueError, match="no reconocido"):
        finops.parse_mib("12 manzanas")


def test_saturacion_descarta_calentamiento_y_calcula_cpu_por_peticion(tmp_path):
    tabla = tabla_de_saturacion(prueba_de_carga(tmp_path))

    assert tabla["users"].tolist() == [1, 4]
    assert tabla["rps"].tolist() == [200.0, 120.0]
    assert tabla.loc[0, "cpu_vcpu"] == pytest.approx(0.88)
    assert tabla.loc[0, "cpu_ms_per_request"] == pytest.approx(4.4)
    assert tabla["memory_mib"].max() == pytest.approx(220)


def test_capacidad_es_el_escalon_mas_cargado_que_cumple_p95(tmp_path):
    tabla = tabla_de_saturacion(prueba_de_carga(tmp_path))

    assert finops.sustainable_capacity(tabla)["users"] == 1
    assert finops.sustainable_capacity(tabla, p95_limit_ms=1000)["users"] == 4
    assert finops.sustainable_capacity(tabla)["peak_rps"] == 200
    with pytest.raises(ValueError, match="cumple"):
        finops.sustainable_capacity(tabla, p95_limit_ms=1)


def test_t3_cobra_creditos_solo_arriba_de_su_linea_base():
    t3_small = finops.INSTANCES["t3.small"]  # línea base de 0.4 vCPU

    assert finops.ec2_hourly_cost(t3_small, 0.3) == pytest.approx(0.0208)
    assert finops.ec2_hourly_cost(t3_small, 0.9) == pytest.approx(0.0208 + 0.5 * 0.05)
    assert finops.ec2_hourly_cost(finops.INSTANCES["c7i.large"], 2) == pytest.approx(0.08925)
    assert finops.ec2_hourly_cost(t3_small, 0.3, spot=True) == pytest.approx(0.008)
    with pytest.raises(ValueError, match="Spot"):
        finops.ec2_hourly_cost(finops.INSTANCES["t3.micro"], 0.1, spot=True)


def test_ec2_rechaza_memoria_que_no_cabe_y_agrega_replicas():
    capacidad = {"rps": 100, "cpu_ms_per_request": 8}
    t3_small = finops.INSTANCES["t3.small"]

    with pytest.raises(ValueError, match="no caben"):
        finops.ec2_cost(t3_small, 1, capacidad, memory_gib=2.6)
    assert finops.ec2_cost(t3_small, 60, capacidad)["replicas"] == 1
    assert finops.ec2_cost(t3_small, 150, capacidad)["replicas"] == 3


def test_cloud_run_conviene_con_poca_demanda_y_no_con_mucha():
    capacidad = {"rps": 125, "cpu_ms_per_request": 8}
    t3_micro = finops.INSTANCES["t3.micro"]
    real = finops.REAL_DEMAND_RPS

    for demanda, cloud_run_gana in [(real, True), (80, False)]:
        cloud_run = finops.cloud_run_cost(demanda, capacidad)["usd_month"]
        ec2 = finops.ec2_cost(t3_micro, demanda, capacidad)["usd_month"]
        assert (cloud_run < ec2) is cloud_run_gana
    assert finops.cloud_run_cost(real, capacidad, free_tier=True)["usd_month"] == 0


def test_lotes_cobran_el_arranque_redondeado_a_100_ms():
    costo = finops.cloud_run_batch_cost(469, cold_start_seconds=2.85, batch_seconds=0.002)

    assert costo["instance_seconds"] == pytest.approx(2.9 * finops.HOURS_PER_MONTH)
    assert costo["predictions_month"] == 469 * finops.HOURS_PER_MONTH


def test_lote_por_estacion_y_tiempos_con_modelo_simulado():
    lote = finops.station_batch(MODELS_DIR / "serving" / "p2" / "input_example.json")

    class Modelo:
        def predict(self, datos):
            return np.zeros((len(datos), 2))

    tiempos = finops.time_predictions(Modelo(), lote, repeats=3)

    assert len(lote) == finops.STATIONS
    assert lote["station_id"].dtype == "int64"
    assert tiempos["rows"] == finops.STATIONS
    assert tiempos["speedup_per_prediction"] > 0


def test_finops_completo_escribe_tablas_figuras_y_resumen(tmp_path):
    carpeta = prueba_de_carga(tmp_path / "carga")
    tiempos = {"rows": 469, "batch_seconds": 0.002, "single_row_seconds": 0.0015}

    resumen = finops.run_finops(
        load_test_dir=carpeta,
        summary_dir=tmp_path / "resumen",
        figure_dir=tmp_path / "figuras",
        output_name="prueba",
        timing=tiempos,
    )

    guardado = json.loads((tmp_path / "resumen" / "prueba.json").read_text(encoding="utf-8"))
    escenarios = pd.read_csv(tmp_path / "resumen" / "prueba_escenarios.csv")
    assert guardado == resumen
    assert guardado["capacity"]["users"] == 1
    assert guardado["cold_start_seconds"] == pytest.approx(2.9)
    assert guardado["price_date"] == finops.PRICE_DATE
    assert len(escenarios) == len(guardado["scenarios"])
    assert (escenarios["usd_per_million"] > 0).all()
    for nombre in ("prueba_saturacion.png", "prueba_costos.png"):
        assert (tmp_path / "figuras" / nombre).stat().st_size > 0
