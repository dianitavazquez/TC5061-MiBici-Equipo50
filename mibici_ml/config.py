import os
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
REFERENCES_DIR = PROJ_ROOT / "references"

# Tracking compartido por los tres modelos
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "mibici-equipo50")

# Defaults shared by the MLflow Project entry points and the Python training code.
RANDOM_SEED = 42

# P1: regresión de demanda
P1_TRIPS_FILE = PROCESSED_DATA_DIR / "mibici_trips.parquet"
P1_MANIFEST_FILE = PROCESSED_DATA_DIR / "manifest.json"
P1_FEATURES_FILE = PROCESSED_DATA_DIR / "p1_features.parquet"
P1_MODEL_FILE = MODELS_DIR / "p1_regression_model.joblib"
P1_COMPARISON_FILE = PROCESSED_DATA_DIR / "p1_model_comparison.csv"
P1_TEST_FRACTION = 0.2
P1_MAX_ITER = 100
P1_LEARNING_RATE = 0.1
P1_MAX_LEAF_NODES = 31
P1_L2_REGULARIZATION = 1.0

# P2: clasificación de presión de demanda
P2_START_DATE = "2025-10-01"
P2_END_DATE = "2026-10-01"
P2_ALPHA = 0.5
P2_TRAIN_FRACTION = 0.8
P2_STATIONS_FILE = "nomenclatura_2026_09.csv"
P2_MAX_ITER = 1000
P2_REGISTERED_MODEL_NAME = "mibici-equipo50-p2-riesgo"
P2_REGISTERED_MODEL_DESCRIPTION = (
    "Clasificador multisalida de la pregunta 2 para estimar, con horizonte de una hora, "
    "el riesgo de presión por falta de bicicletas y por falta de anclajes en estaciones "
    "MiBici. El alias champion identifica la versión aprobada para producción."
)
P2_SEARCH_MODELS = {
    "candidato-1": {"class_weight": "balanced", "c": 1.0},
    "candidato-2": {"class_weight": {0: 1.0, 1: 2.0}, "c": 1.0},
    "candidato-3": {"class_weight": {0: 1.0, 1: 5.0}, "c": 1.0},
}

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
SELECTED_SOURCE_FILES = [*CLUSTERING_TRIP_FILES, CLUSTERING_STATIONS_FILE]

CLUSTERING_START_DATE = "2025-09-01"
CLUSTERING_END_DATE = "2026-09-30"

# Reglas de limpieza de viajes
CLUSTERING_MIN_TRIP_SECONDS = 60  # misma estación y menos de esto: arranque falso
CLUSTERING_MAX_TRIP_HOURS = 3  # más de esto: bici mal anclada o error de registro

# Salidas de la ingesta
CLUSTERING_TRIPS_FILE = INTERIM_DATA_DIR / "viajes_clustering.parquet"
CLUSTERING_STATIONS_CLEAN_FILE = INTERIM_DATA_DIR / "estaciones_clustering.parquet"
CLUSTERING_MANIFEST_FILE = REFERENCES_DIR / "manifest_clustering.json"

# Perfil de uso por estación (features)
CLUSTERING_MIN_ACTIVE_DAYS = 30  # menos días con viajes que esto: perfil poco confiable
CLUSTERING_FEATURES_FILE = PROCESSED_DATA_DIR / "perfil_estaciones_clustering.parquet"

# Experimentos en MLflow (base de datos local, se crea en la raíz del repo)
CLUSTERING_MLFLOW_URI = MLFLOW_TRACKING_URI
CLUSTERING_MLFLOW_EXPERIMENT = MLFLOW_EXPERIMENT

CLUSTERING_SEED = RANDOM_SEED
CLUSTERING_K = 4
CLUSTERING_N_INIT = 10

# P3: experimentación y Model Registry (segunda entrega)
CLUSTERING_SEARCH_KS = [3, 4, 5, 6, 7, 8]  # búsqueda de k con nested runs
CLUSTERING_STABILITY_RUNS = 20  # repeticiones con submuestras para medir estabilidad
CLUSTERING_STABILITY_FRACTION = 0.8  # porcentaje de estaciones en cada submuestra
# Baseline y candidatos: algoritmo, conjunto de variables e hiperparámetros
CLUSTERING_CANDIDATES = {
    "baseline": {"algoritmo": "kmeans", "variables": "basico", "k": 4, "n_init": 10},
    "candidato-1": {"algoritmo": "kmeans", "variables": "uso", "k": 4, "n_init": 10},
    "candidato-2": {
        "algoritmo": "kmeans",
        "variables": "uso_sin_mediodia",
        "k": 4,
        "n_init": 10,
    },
    "candidato-3": {
        "algoritmo": "gmm",
        "variables": "uso",
        "k": 4,
        "n_init": 3,
        "covariance_type": "diag",
    },
}
CLUSTERING_REGISTERED_MODEL = "mibici-equipo50-p3-grupos"
