"""Ingesta reproducible de los datos de MiBici para el clustering de estaciones (P3).

Lee los archivos de data/raw, normaliza columnas, codificación y fechas, limpia los
viajes y guarda el resultado en Parquet. También genera un manifiesto con el número de
registros y el SHA-256 de cada archivo, que identifica la versión exacta de los datos.

Uso:
    python -m mibici_ml.dataset_clustering
"""

import hashlib
import json
import re
import unicodedata

from loguru import logger
import pandas as pd
import typer

from mibici_ml.config import (
    CLUSTERING_END_DATE,
    CLUSTERING_MANIFEST_FILE,
    CLUSTERING_MAX_TRIP_HOURS,
    CLUSTERING_MIN_TRIP_SECONDS,
    CLUSTERING_START_DATE,
    CLUSTERING_STATIONS_CLEAN_FILE,
    CLUSTERING_STATIONS_FILE,
    CLUSTERING_TRIP_FILES,
    CLUSTERING_TRIPS_FILE,
    RAW_DATA_DIR,
)

app = typer.Typer()

# Columnas de los viajes que se usan en P3, con su nombre normalizado.
# Usuario_Id, Genero y Año_de_nacimiento no se cargan: no aportan al patrón de uso de una
# estación y son datos personales.
COLUMNAS_VIAJES = {
    "viaje_id": "viaje_id",
    "inicio_del_viaje": "inicio",
    "fin_del_viaje": "fin",
    "origen_id": "origen_id",
    "destino_id": "destino_id",
}
COLUMNAS_ESTACIONES = {
    "id": "estacion_id",
    "name": "nombre",
    "latitude": "latitud",
    "longitude": "longitud",
    "dpcapacity": "capacidad",
}
FORMATO_FECHA = "%Y-%m-%d %H:%M:%S"


def revisar_archivos(nombres):
    """Confirma que todos los archivos estén en data/raw antes de empezar."""
    faltantes = [nombre for nombre in nombres if not (RAW_DATA_DIR / nombre).exists()]
    if faltantes:
        raise FileNotFoundError(
            f"Faltan estos archivos en {RAW_DATA_DIR}: {', '.join(faltantes)}. "
            "Descárgalos de https://www.mibici.net/es/datos-abiertos"
        )


def calcular_sha256(ruta):
    """Huella del archivo: si cambia un solo byte, cambia el hash."""
    huella = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            huella.update(bloque)
    return huella.hexdigest()


def detectar_codificacion(ruta):
    """Casi todos los archivos vienen en UTF-8, pero algunos meses (como enero 2026) en cp1252."""
    try:
        ruta.read_bytes().decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1252"


def normalizar_columna(nombre):
    """Quita acentos, espacios y mayúsculas, por ejemplo 'Año_de_nacimiento' a 'ano_de_nacimiento'.

    Así los archivos se leen igual aunque el nombre de una columna cambie entre años.
    """
    sin_acentos = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", sin_acentos.strip().lower()).strip("_")


def leer_viajes(ruta):
    """Lee un archivo mensual solo con las columnas necesarias y con nombres normalizados."""
    codificacion = detectar_codificacion(ruta)
    viajes = pd.read_csv(
        ruta,
        encoding=codificacion,
        usecols=lambda columna: normalizar_columna(columna) in COLUMNAS_VIAJES,
        dtype=str,
    )
    viajes.columns = [COLUMNAS_VIAJES[normalizar_columna(c)] for c in viajes.columns]
    faltantes = set(COLUMNAS_VIAJES.values()) - set(viajes.columns)
    if faltantes:
        raise ValueError(f"{ruta.name} no tiene las columnas {sorted(faltantes)}")
    return viajes, codificacion


def convertir_tipos(viajes):
    """Convierte fechas a datetime e identificadores a enteros; lo que no se pueda queda nulo."""
    viajes = viajes.copy()
    for columna in ["inicio", "fin"]:
        viajes[columna] = pd.to_datetime(viajes[columna], format=FORMATO_FECHA, errors="coerce")
    for columna in ["viaje_id", "origen_id", "destino_id"]:
        viajes[columna] = pd.to_numeric(viajes[columna], errors="coerce").astype("Int64")
    return viajes


def _filtrar(viajes, mantener, motivo, resumen):
    """Quita las filas que no cumplen una regla y anota cuántas fueron."""
    resumen[motivo] = int((~mantener).sum())
    return viajes[mantener]


def limpiar_viajes(viajes, ids_estaciones):
    """Aplica las reglas de limpieza y regresa los viajes limpios y un resumen de lo que se quitó."""
    resumen = {"registros_originales": len(viajes)}
    viajes = _filtrar(viajes, ~viajes.duplicated(), "duplicados", resumen)
    viajes = _filtrar(viajes, viajes.notna().all(axis=1), "valores_invalidos", resumen)

    inicio_rango = pd.Timestamp(CLUSTERING_START_DATE)
    fin_rango = pd.Timestamp(CLUSTERING_END_DATE) + pd.Timedelta(days=1)
    en_rango = (viajes["inicio"] >= inicio_rango) & (viajes["inicio"] < fin_rango)
    viajes = _filtrar(viajes, en_rango, "fuera_de_rango", resumen)

    duracion = (viajes["fin"] - viajes["inicio"]).dt.total_seconds()
    misma_estacion = viajes["origen_id"] == viajes["destino_id"]
    viajes = _filtrar(viajes, duracion >= 0, "duracion_negativa", resumen)
    duracion = duracion[viajes.index]
    misma_estacion = misma_estacion[viajes.index]

    arranque_falso = misma_estacion & (duracion < CLUSTERING_MIN_TRIP_SECONDS)
    viajes = _filtrar(viajes, ~arranque_falso, "arranque_falso", resumen)
    duracion = duracion[viajes.index]

    muy_largo = duracion > CLUSTERING_MAX_TRIP_HOURS * 3600
    viajes = _filtrar(viajes, ~muy_largo, "duracion_mayor_al_maximo", resumen)

    origen_conocido = viajes["origen_id"].isin(ids_estaciones)
    destino_conocido = viajes["destino_id"].isin(ids_estaciones)
    conocidas = origen_conocido & destino_conocido
    viajes = _filtrar(viajes, conocidas, "estacion_sin_nomenclatura", resumen)

    limpios = pd.DataFrame(
        {
            "inicio": viajes["inicio"],
            "fin": viajes["fin"],
            "origen_id": viajes["origen_id"].astype("int32"),
            "destino_id": viajes["destino_id"].astype("int32"),
            "duracion_min": ((viajes["fin"] - viajes["inicio"]).dt.total_seconds() / 60).round(2),
        }
    ).reset_index(drop=True)
    resumen["registros_limpios"] = len(limpios)
    return limpios, resumen


def limpiar_estaciones(ruta):
    """Normaliza la nomenclatura y agrega la zona (municipio) y si la estación sigue activa.

    La zona sale del prefijo del nombre (GDL Guadalajara, ZPN Zapopan, TLQ Tlaquepaque) y
    las estaciones retiradas tienen el prefijo "ex-".
    """
    estaciones = pd.read_csv(ruta, encoding=detectar_codificacion(ruta))
    estaciones.columns = [normalizar_columna(c) for c in estaciones.columns]
    estaciones = estaciones.rename(columns=COLUMNAS_ESTACIONES)
    estaciones["zona"] = estaciones["nombre"].str.extract(r"(GDL|ZPN|TLQ)", expand=False)
    estaciones["activa"] = ~estaciones["nombre"].str.startswith("ex-")
    columnas = ["estacion_id", "nombre", "zona", "latitud", "longitud", "capacidad", "activa"]
    return estaciones[columnas]


def describir_archivo(ruta, registros, codificacion):
    """Datos de un archivo para el manifiesto."""
    return {
        "archivo": ruta.name,
        "bytes": ruta.stat().st_size,
        "sha256": calcular_sha256(ruta),
        "registros": registros,
        "codificacion": codificacion,
    }


def crear_manifiesto(archivos, resumen_limpieza):
    """La versión de los datos es un hash corto de las huellas de todos los archivos."""
    huellas = "".join(sorted(archivo["sha256"] for archivo in archivos))
    return {
        "pregunta": "P3",
        "fuente": "https://www.mibici.net/es/datos-abiertos",
        "rango": {"inicio": CLUSTERING_START_DATE, "fin": CLUSTERING_END_DATE},
        "version_datos": hashlib.sha256(huellas.encode()).hexdigest()[:12],
        "archivos": archivos,
        "limpieza": resumen_limpieza,
    }


@app.command()
def main():
    """Lee, limpia y guarda los datos de P3, y escribe su manifiesto."""
    revisar_archivos([*CLUSTERING_TRIP_FILES, CLUSTERING_STATIONS_FILE])

    ruta_estaciones = RAW_DATA_DIR / CLUSTERING_STATIONS_FILE
    estaciones = limpiar_estaciones(ruta_estaciones)
    info_archivos = [
        describir_archivo(ruta_estaciones, len(estaciones), detectar_codificacion(ruta_estaciones))
    ]

    meses = []
    for nombre in CLUSTERING_TRIP_FILES:
        ruta = RAW_DATA_DIR / nombre
        viajes_mes, codificacion = leer_viajes(ruta)
        logger.info(f"{nombre}: {len(viajes_mes):,} viajes ({codificacion})")
        meses.append(viajes_mes)
        info_archivos.append(describir_archivo(ruta, len(viajes_mes), codificacion))

    viajes = convertir_tipos(pd.concat(meses, ignore_index=True))
    viajes, resumen = limpiar_viajes(viajes, set(estaciones["estacion_id"]))
    for motivo, cantidad in resumen.items():
        logger.info(f"  {motivo}: {cantidad:,}")

    CLUSTERING_TRIPS_FILE.parent.mkdir(parents=True, exist_ok=True)
    viajes.to_parquet(CLUSTERING_TRIPS_FILE, index=False)
    estaciones.to_parquet(CLUSTERING_STATIONS_CLEAN_FILE, index=False)

    manifiesto = crear_manifiesto(info_archivos, resumen)
    CLUSTERING_MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLUSTERING_MANIFEST_FILE.write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.success(
        f"Datos guardados en {CLUSTERING_TRIPS_FILE.parent} | "
        f"versión de datos {manifiesto['version_datos']}"
    )


if __name__ == "__main__":
    app()
