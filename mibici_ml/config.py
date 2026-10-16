from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# ---------------------------------------------------------------------------
# P3: Clustering de estaciones
# ---------------------------------------------------------------------------
REFERENCES_DIR = PROJ_ROOT / "references"

# Archivos que deben estar en data/raw (descargados del portal de datos abiertos de MiBici)
# Viajes de septiembre 2025 a septiembre 2026 (mismo rango que el resto del equipo)
CLUSTERING_TRIP_FILES = [
    "datos_abiertos_2025_09.csv",
    "datos_abiertos_2025_10.csv",
    "datos_abiertos_2025_11.csv",
    "datos_abiertos_2025_12.csv",
    "datos_abiertos_2026_01.csv",
    "datos_abiertos_2026_02.csv",
    "datos_abiertos_2026_03.csv",
    "datos_abiertos_2026_04.csv",
    "datos_abiertos_2026_05.csv",
    "datos_abiertos_2026_06.csv",
    "datos_abiertos_2026_07.csv",
    "datos_abiertos_2026_08.csv",
    "datos_abiertos_2026_09.csv",
]
CLUSTERING_STATIONS_FILE = "nomenclatura_2026_09.csv"

CLUSTERING_START_DATE = "2025-09-01"
CLUSTERING_END_DATE = "2026-09-30"

# Reglas de limpieza de viajes
CLUSTERING_MIN_TRIP_SECONDS = 60  # misma estación y menos de esto: arranque falso
CLUSTERING_MAX_TRIP_HOURS = 3  # más de esto: bici mal anclada o error de registro

# Salidas de la ingesta
CLUSTERING_TRIPS_FILE = INTERIM_DATA_DIR / "viajes_clustering.parquet"
CLUSTERING_STATIONS_CLEAN_FILE = INTERIM_DATA_DIR / "estaciones_clustering.parquet"
CLUSTERING_MANIFEST_FILE = REFERENCES_DIR / "manifest_clustering.json"

CLUSTERING_SEED = 42
