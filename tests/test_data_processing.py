"""Pruebas unitarias del procesamiento de datos (ingesta y limpieza en dataset.py).

Cada prueba arma archivos pequeños en una carpeta temporal, así que no necesita los datos
reales de MiBici ni escribe en data/ o references/.
"""

import json

import pandas as pd
import pytest

from mibici_ml import dataset
from mibici_ml.config import (
    CLUSTERING_STATIONS_FILE,
    CLUSTERING_TRIP_FILES,
    SELECTED_SOURCE_FILES,
)
from mibici_ml.dataset import (
    clean_p3_stations,
    clean_p3_trips,
    convert_p3_types,
    detect_encoding,
    ingest,
    normalize_column,
    prepare_p3,
    read_p3_trips,
    sha256_file,
)

COLUMNAS_LIMPIAS = ["inicio", "fin", "origen_id", "destino_id", "duracion_min"]
ESTACIONES = {1, 2, 3}


def viajes_p3(filas):
    """Arma viajes con los tipos que usa la limpieza de P3.

    Cada fila es (viaje_id, inicio, fin, origen_id, destino_id) como texto, igual que en los
    CSV originales.
    """
    columnas = ["viaje_id", "inicio", "fin", "origen_id", "destino_id"]
    return convert_p3_types(pd.DataFrame(filas, columns=columnas, dtype=str))


def escribir_viajes_crudos(carpeta, filas_por_mes=None):
    """Escribe los 13 CSV mensuales con el formato original de MiBici.

    filas_por_mes permite cambiar las filas de un archivo; los demás llevan un viaje válido
    con un Viaje_Id distinto por mes.
    """
    filas_por_mes = filas_por_mes or {}
    for numero, nombre in enumerate(CLUSTERING_TRIP_FILES):
        filas = filas_por_mes.get(
            nombre,
            [[numero + 1, 10, "M", 1990, "2025-10-01 08:00:00", "2025-10-01 08:12:00", 1, 2]],
        )
        pd.DataFrame(
            filas,
            columns=[
                "Viaje_Id",
                "Usuario_Id",
                "Genero",
                "Año_de_nacimiento",
                "Inicio_del_viaje",
                "Fin_del_viaje",
                "Origen_Id",
                "Destino_Id",
            ],
        ).to_csv(carpeta / nombre, index=False)


def escribir_nomenclatura(carpeta):
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["(GDL-001) Centro", "(ZPN-002) Zapopan", "ex-(TLQ-003) Retirada"],
            "latitude": [20.67, 20.72, 20.64],
            "longitude": [-103.35, -103.39, -103.31],
            "dpcapacity": [15, 19, 0],
        }
    ).to_csv(carpeta / CLUSTERING_STATIONS_FILE, index=False)


# ---------------------------------------------------------------------------
# normalize_column y detect_encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("original", "esperado"),
    [
        ("Año_de_nacimiento", "ano_de_nacimiento"),
        (" Inicio del viaje ", "inicio_del_viaje"),
        ("Origen-Id", "origen_id"),
        ("DPCapacity", "dpcapacity"),
        ("__Viaje  Id__", "viaje_id"),
        ("", ""),
    ],
)
def test_normalizar_columna_quita_acentos_espacios_y_mayusculas(original, esperado):
    assert normalize_column(original) == esperado


def test_detectar_codificacion_distingue_utf8_y_cp1252(tmp_path):
    utf8 = tmp_path / "utf8.csv"
    cp1252 = tmp_path / "cp1252.csv"
    utf8.write_text("name\nC. Niños Héroes\n", encoding="utf-8")
    cp1252.write_text("name\nC. Niños Héroes\n", encoding="cp1252")

    assert detect_encoding(utf8) == "utf-8"
    assert detect_encoding(cp1252) == "cp1252"


# ---------------------------------------------------------------------------
# read_p3_trips y convert_p3_types
# ---------------------------------------------------------------------------


def test_leer_viajes_no_carga_datos_personales(tmp_path):
    escribir_viajes_crudos(tmp_path)

    viajes, codificacion = read_p3_trips(tmp_path / CLUSTERING_TRIP_FILES[0])

    assert list(viajes.columns) == ["viaje_id", "inicio", "fin", "origen_id", "destino_id"]
    assert codificacion == "utf-8"


def test_leer_viajes_acepta_otros_nombres_de_columnas_y_cp1252(tmp_path):
    archivo = tmp_path / "viajes.csv"
    pd.DataFrame(
        {
            "Viaje Id": [1],
            "Inicio del viaje": ["2025-10-01 08:00:00"],
            "FIN_DEL_VIAJE": ["2025-10-01 08:10:00"],
            "Origen_Id": [1],
            "Destino_Id": [2],
            "Género": ["F"],
        }
    ).to_csv(archivo, index=False, encoding="cp1252")

    viajes, codificacion = read_p3_trips(archivo)

    assert codificacion == "cp1252"
    assert list(viajes.columns) == ["viaje_id", "inicio", "fin", "origen_id", "destino_id"]


def test_leer_viajes_falla_si_falta_una_columna(tmp_path):
    archivo = tmp_path / "viajes.csv"
    pd.DataFrame({"Viaje_Id": [1], "Inicio_del_viaje": ["2025-10-01 08:00:00"]}).to_csv(
        archivo, index=False
    )

    with pytest.raises(ValueError, match="no tiene las columnas"):
        read_p3_trips(archivo)


def test_convertir_tipos_deja_nulos_los_valores_invalidos():
    viajes = viajes_p3(
        [
            ["1", "2025-10-01 08:00:00", "2025-10-01 08:10:00", "1", "2"],
            ["x", "2025-13-01 08:00:00", "no es fecha", "uno", "2"],
        ]
    )

    assert viajes["inicio"].iloc[0] == pd.Timestamp("2025-10-01 08:00:00")
    assert viajes.iloc[1][["viaje_id", "inicio", "fin", "origen_id"]].isna().all()
    assert str(viajes["origen_id"].dtype) == "Int64"


# ---------------------------------------------------------------------------
# clean_p3_trips
# ---------------------------------------------------------------------------


def test_limpiar_viajes_quita_un_caso_por_cada_regla():
    valido = ["1", "2025-10-01 08:00:00", "2025-10-01 08:10:00", "1", "2"]
    viajes = viajes_p3(
        [
            valido,
            valido,  # duplicado
            ["2", "no es fecha", "2025-10-01 08:10:00", "1", "2"],  # valor inválido
            ["3", "2025-08-31 23:00:00", "2025-08-31 23:10:00", "1", "2"],  # fuera de rango
            ["4", "2025-10-01 08:10:00", "2025-10-01 08:00:00", "1", "2"],  # duración negativa
            ["5", "2025-10-01 08:00:00", "2025-10-01 08:00:30", "1", "1"],  # arranque falso
            ["6", "2025-10-01 08:00:00", "2025-10-01 11:00:01", "1", "2"],  # más de 3 h
            ["7", "2025-10-01 08:00:00", "2025-10-01 08:10:00", "1", "99"],  # sin nomenclatura
        ]
    )

    limpios, resumen = clean_p3_trips(viajes, ESTACIONES)

    assert resumen == {
        "registros_originales": 8,
        "duplicados": 1,
        "valores_invalidos": 1,
        "fuera_de_rango": 1,
        "duracion_negativa": 1,
        "arranque_falso": 1,
        "duracion_mayor_al_maximo": 1,
        "estacion_sin_nomenclatura": 1,
        "registros_limpios": 1,
    }
    assert len(limpios) == 1


def test_limpiar_viajes_conserva_los_casos_en_el_limite():
    viajes = viajes_p3(
        [
            ["1", "2025-10-01 08:00:00", "2025-10-01 08:01:00", "1", "1"],  # misma, 60 s
            ["2", "2025-10-01 08:00:00", "2025-10-01 08:00:10", "1", "2"],  # otra, 10 s
            ["3", "2025-10-01 08:00:00", "2025-10-01 11:00:00", "1", "2"],  # justo 3 h
            ["4", "2025-09-01 00:00:00", "2025-09-01 00:10:00", "1", "2"],  # primer instante
            ["5", "2026-09-30 23:59:59", "2026-10-01 00:09:59", "1", "2"],  # último instante
            ["6", "2026-10-01 00:00:00", "2026-10-01 00:10:00", "1", "2"],  # ya fuera
        ]
    )

    limpios, resumen = clean_p3_trips(viajes, ESTACIONES)

    assert len(limpios) == 5
    assert resumen["fuera_de_rango"] == 1
    assert resumen["arranque_falso"] == 0
    assert resumen["duracion_mayor_al_maximo"] == 0


def test_limpiar_viajes_entrega_columnas_tipos_y_duracion_en_minutos():
    viajes = viajes_p3([["1", "2025-10-01 08:00:00", "2025-10-01 08:01:30", "1", "2"]])

    limpios, _ = clean_p3_trips(viajes, ESTACIONES)

    assert list(limpios.columns) == COLUMNAS_LIMPIAS
    assert str(limpios["origen_id"].dtype) == "int32"
    assert str(limpios["destino_id"].dtype) == "int32"
    assert limpios["duracion_min"].iloc[0] == 1.5


def test_limpiar_viajes_no_modifica_la_tabla_original():
    viajes = viajes_p3(
        [
            ["1", "2025-10-01 08:00:00", "2025-10-01 08:10:00", "1", "2"],
            ["2", "2025-10-01 08:00:00", "2025-10-01 08:00:20", "1", "1"],
        ]
    )
    copia = viajes.copy()

    clean_p3_trips(viajes, ESTACIONES)

    pd.testing.assert_frame_equal(viajes, copia)


def test_limpiar_viajes_acepta_una_tabla_vacia():
    limpios, resumen = clean_p3_trips(viajes_p3([]), ESTACIONES)

    assert limpios.empty
    assert list(limpios.columns) == COLUMNAS_LIMPIAS
    assert resumen["registros_originales"] == 0
    assert resumen["registros_limpios"] == 0


# ---------------------------------------------------------------------------
# clean_p3_stations
# ---------------------------------------------------------------------------


def test_limpiar_estaciones_agrega_zona_y_marca_las_retiradas(tmp_path):
    escribir_nomenclatura(tmp_path)

    estaciones = clean_p3_stations(tmp_path / CLUSTERING_STATIONS_FILE)

    assert estaciones["zona"].tolist() == ["GDL", "ZPN", "TLQ"]
    assert estaciones["activa"].tolist() == [True, True, False]
    assert str(estaciones["estacion_id"].dtype) == "int32"


def test_limpiar_estaciones_deja_nula_una_capacidad_no_numerica(tmp_path):
    archivo = tmp_path / "nomenclatura.csv"
    pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["(GDL-001) Centro", "Estación sin prefijo"],
            "latitude": ["20.67", "20.70"],
            "longitude": ["-103.35", "-103.36"],
            "dpcapacity": ["15", "N/A"],
        }
    ).to_csv(archivo, index=False)

    estaciones = clean_p3_stations(archivo)

    assert estaciones["capacidad"].iloc[0] == 15
    assert pd.isna(estaciones["capacidad"].iloc[1])
    assert pd.isna(estaciones["zona"].iloc[1])


def test_limpiar_estaciones_falla_si_falta_una_columna(tmp_path):
    archivo = tmp_path / "nomenclatura.csv"
    pd.DataFrame({"id": [1], "name": ["(GDL-001) Centro"]}).to_csv(archivo, index=False)

    with pytest.raises(ValueError, match="no tiene las columnas"):
        clean_p3_stations(archivo)


# ---------------------------------------------------------------------------
# Versionado y reproducibilidad: sha256_file y prepare_p3
# ---------------------------------------------------------------------------


def test_sha256_es_igual_para_el_mismo_contenido_y_cambia_con_un_byte(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_bytes(b"Viaje_Id\n1\n")
    b.write_bytes(b"Viaje_Id\n1\n")

    assert sha256_file(a) == sha256_file(b)
    b.write_bytes(b"Viaje_Id\n2\n")
    assert sha256_file(a) != sha256_file(b)


@pytest.fixture
def salidas_p3(tmp_path, monkeypatch):
    """Redirige las salidas de prepare_p3 a la carpeta temporal."""
    salidas = {
        "viajes": tmp_path / "salida" / "viajes.parquet",
        "estaciones": tmp_path / "salida" / "estaciones.parquet",
        "manifiesto": tmp_path / "salida" / "manifest.json",
    }
    monkeypatch.setattr(dataset, "CLUSTERING_TRIPS_FILE", salidas["viajes"])
    monkeypatch.setattr(dataset, "CLUSTERING_STATIONS_CLEAN_FILE", salidas["estaciones"])
    monkeypatch.setattr(dataset, "CLUSTERING_MANIFEST_FILE", salidas["manifiesto"])
    return salidas


def test_preparar_p3_es_reproducible_y_versiona_los_datos(tmp_path, salidas_p3):
    crudos = tmp_path / "raw"
    crudos.mkdir()
    escribir_viajes_crudos(crudos)
    escribir_nomenclatura(crudos)

    prepare_p3(crudos)
    primero = json.loads(salidas_p3["manifiesto"].read_text(encoding="utf-8"))
    viajes_primero = pd.read_parquet(salidas_p3["viajes"])
    prepare_p3(crudos)
    segundo = json.loads(salidas_p3["manifiesto"].read_text(encoding="utf-8"))

    assert primero["version_datos"] == segundo["version_datos"]
    pd.testing.assert_frame_equal(viajes_primero, pd.read_parquet(salidas_p3["viajes"]))
    assert len(primero["archivos"]) == len(SELECTED_SOURCE_FILES)
    assert primero["limpieza"]["registros_limpios"] == len(CLUSTERING_TRIP_FILES)

    with (crudos / CLUSTERING_TRIP_FILES[0]).open("a", encoding="utf-8") as archivo:
        archivo.write("\n")
    prepare_p3(crudos)
    tercero = json.loads(salidas_p3["manifiesto"].read_text(encoding="utf-8"))
    assert tercero["version_datos"] != primero["version_datos"]


def test_preparar_p3_falla_si_falta_un_archivo(tmp_path, salidas_p3):
    escribir_viajes_crudos(tmp_path)

    with pytest.raises(FileNotFoundError, match=CLUSTERING_STATIONS_FILE):
        prepare_p3(tmp_path)


# ---------------------------------------------------------------------------
# ingest (P1)
# ---------------------------------------------------------------------------


def test_ingesta_quita_viajes_duplicados_dentro_de_un_mes(tmp_path):
    fila = [100, 10, "M", 1990, "2025-10-01 08:00:00", "2025-10-01 08:12:00", 1, 2]
    escribir_viajes_crudos(tmp_path, {CLUSTERING_TRIP_FILES[0]: [fila, fila]})
    salida, manifiesto = tmp_path / "viajes.parquet", tmp_path / "manifest.json"

    ingest(tmp_path, salida, manifiesto)

    resumen = json.loads(manifiesto.read_text(encoding="utf-8"))["summary"]
    assert resumen["duplicate_rows_removed"] == 1
    assert len(pd.read_parquet(salida)) == len(CLUSTERING_TRIP_FILES)


def test_ingesta_falla_con_fechas_invalidas(tmp_path):
    fila = [100, 10, "M", 1990, "no es fecha", "2025-10-01 08:12:00", 1, 2]
    escribir_viajes_crudos(tmp_path, {CLUSTERING_TRIP_FILES[0]: [fila]})

    with pytest.raises(ValueError, match="invalid timestamps"):
        ingest(tmp_path, tmp_path / "viajes.parquet", tmp_path / "manifest.json")


def test_ingesta_falla_si_un_viaje_se_repite_entre_meses(tmp_path):
    fila = [1, 10, "M", 1990, "2025-10-01 08:00:00", "2025-10-01 08:12:00", 1, 2]
    escribir_viajes_crudos(
        tmp_path, {CLUSTERING_TRIP_FILES[0]: [fila], CLUSTERING_TRIP_FILES[1]: [fila]}
    )

    with pytest.raises(ValueError, match="Repeated Viaje_Id"):
        ingest(tmp_path, tmp_path / "viajes.parquet", tmp_path / "manifest.json")
