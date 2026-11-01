"""Dataset quality and repository security checks.

The data checks use generated local artifacts when they are available. They are skipped in
a clean clone until the ingestion and feature stages have been executed.
"""

from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype
import pytest

from mibici_ml.config import (
    CLUSTERING_FEATURES_FILE,
    CLUSTERING_TRIPS_FILE,
    P1_FEATURES_FILE,
    P1_TRIPS_FILE,
    P2_STATIONS_FILE,
    RAW_DATA_DIR,
)
from mibici_ml.dataset import TRIP_COLUMNS, read_csv_compatible
from mibici_ml.features import build_station_hour_dataset
from mibici_ml.modeling.train import NUMERIC_FEATURES, TARGETS, chronological_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
P1_REQUIRED_COLUMNS = {
    "station_id",
    "timestamp",
    "target_departures",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "departures_lag_1",
    "departures_lag_24",
    "departures_lag_168",
    "departures_mean_24",
    "departures_mean_168",
}
P2_REQUIRED_COLUMNS = {
    "station_id",
    "hour",
    *NUMERIC_FEATURES,
    *TARGETS,
}
P2_DRIFT_COLUMNS = [
    "departures_lag_1h",
    "arrivals_lag_1h",
    "net_departures_lag_1h",
    "departures_mean_24h",
    "arrivals_mean_24h",
    *TARGETS,
]
P3_PROPORTION_COLUMNS = [
    "prop_manana",
    "prop_mediodia",
    "prop_tarde",
    "prop_noche",
    "prop_fin_semana",
    "prop_redondo",
]
DRIFT_COLUMNS = [
    "target_departures",
    "departures_lag_1",
    "departures_lag_24",
    "departures_lag_168",
    "departures_mean_24",
    "departures_mean_168",
]


def read_parquet_or_skip(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"Falta {path.name}; ejecuta primero las etapas de ingesta y features.")
    return pd.read_parquet(path, columns=columns)


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calculate PSI using bins learned exclusively from the reference partition."""
    edges = np.unique(reference.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(edges) < 3:
        edges = np.array([-np.inf, float(reference.median()), np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf

    reference_ratio = np.histogram(reference, bins=edges)[0] / len(reference)
    current_ratio = np.histogram(current, bins=edges)[0] / len(current)
    reference_ratio = np.clip(reference_ratio, 1e-6, None)
    current_ratio = np.clip(current_ratio, 1e-6, None)
    return float(
        np.sum((reference_ratio - current_ratio) * np.log(reference_ratio / current_ratio))
    )


@pytest.fixture(scope="module")
def p1_features() -> pd.DataFrame:
    return read_parquet_or_skip(P1_FEATURES_FILE)


@pytest.fixture(scope="module")
def p2_features() -> pd.DataFrame:
    """Build a representative real P2 month because this dataset is not persisted."""
    stations_path = RAW_DATA_DIR / P2_STATIONS_FILE
    if not P1_TRIPS_FILE.exists() or not stations_path.exists():
        pytest.skip("Faltan los insumos locales de P2; ejecuta primero la etapa de ingesta.")

    start = pd.Timestamp("2025-10-01")
    end = pd.Timestamp("2025-11-01")
    trips = pd.read_parquet(
        P1_TRIPS_FILE,
        columns=TRIP_COLUMNS,
        filters=[
            ("Inicio_del_viaje", ">=", start),
            ("Inicio_del_viaje", "<", end),
        ],
    )
    valid = trips[TRIP_COLUMNS].notna().all(axis=1)
    valid &= trips["Fin_del_viaje"].ge(trips["Inicio_del_viaje"])
    valid &= trips["Origen_Id"].ne(trips["Destino_Id"])
    trips = trips.loc[valid].copy()
    stations = read_csv_compatible(stations_path)
    return build_station_hour_dataset(trips, stations)


def test_p1_feature_dataset_has_valid_schema_and_values(p1_features):
    missing = P1_REQUIRED_COLUMNS - set(p1_features.columns)
    assert not missing, f"Columnas faltantes en P1: {sorted(missing)}"
    assert is_datetime64_any_dtype(p1_features["timestamp"])
    assert is_integer_dtype(p1_features["station_id"])
    assert all(is_numeric_dtype(p1_features[column]) for column in DRIFT_COLUMNS)
    assert not p1_features[list(P1_REQUIRED_COLUMNS)].isna().any().any()
    assert not p1_features.duplicated(["station_id", "timestamp"]).any()
    assert np.isfinite(p1_features[DRIFT_COLUMNS].to_numpy()).all()
    assert p1_features["target_departures"].ge(0).all()
    assert p1_features["hour"].between(0, 23).all()
    assert p1_features["day_of_week"].between(0, 6).all()
    assert p1_features["month"].between(1, 12).all()
    assert set(p1_features["is_weekend"].unique()) <= {0, 1}


def test_p1_train_test_partitions_have_no_material_drift(p1_features):
    timestamps = np.sort(p1_features["timestamp"].unique())
    cutoff = pd.Timestamp(timestamps[int(len(timestamps) * 0.8)])
    train = p1_features[p1_features["timestamp"] < cutoff]
    test = p1_features[p1_features["timestamp"] >= cutoff]

    assert not train.empty and not test.empty
    assert train["timestamp"].max() < test["timestamp"].min()
    drift = {
        column: population_stability_index(train[column], test[column]) for column in DRIFT_COLUMNS
    }
    assert max(drift.values()) < 0.25, f"Drift PSI material detectado: {drift}"


def test_p2_station_hour_dataset_has_valid_schema_and_values(p2_features):
    missing = P2_REQUIRED_COLUMNS - set(p2_features.columns)
    assert not missing, f"Columnas faltantes en P2: {sorted(missing)}"
    assert is_datetime64_any_dtype(p2_features["hour"])
    assert is_integer_dtype(p2_features["station_id"])
    assert all(is_numeric_dtype(p2_features[column]) for column in NUMERIC_FEATURES + TARGETS)
    assert not p2_features[list(P2_REQUIRED_COLUMNS)].isna().any().any()
    assert not p2_features.duplicated(["station_id", "hour"]).any()
    assert np.isfinite(p2_features[NUMERIC_FEATURES + TARGETS].to_numpy()).all()
    assert p2_features["capacity"].gt(0).all()
    assert set(p2_features["is_weekend"].unique()) <= {0, 1}
    assert (
        p2_features[
            ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos"]
        ]
        .apply(lambda values: values.between(-1, 1))
        .all()
        .all()
    )
    for target in TARGETS:
        assert set(p2_features[target].unique()) <= {0, 1}


def test_p2_train_test_partitions_have_no_material_drift(p2_features):
    train, test, cutoff = chronological_split(p2_features, train_fraction=0.8)

    assert train["hour"].max() < cutoff == test["hour"].min()
    drift = {
        column: population_stability_index(train[column], test[column])
        for column in P2_DRIFT_COLUMNS
    }
    assert max(drift.values()) < 0.25, f"Drift PSI material detectado en P2: {drift}"


def test_p3_trip_dataset_has_valid_types_missing_values_and_duplicate_rate():
    trips = read_parquet_or_skip(CLUSTERING_TRIPS_FILE)

    assert list(trips.columns) == [
        "inicio",
        "fin",
        "origen_id",
        "destino_id",
        "duracion_min",
    ]
    assert is_datetime64_any_dtype(trips["inicio"])
    assert is_datetime64_any_dtype(trips["fin"])
    assert is_integer_dtype(trips["origen_id"])
    assert is_integer_dtype(trips["destino_id"])
    assert is_numeric_dtype(trips["duracion_min"])
    assert not trips.isna().any().any()
    assert trips["fin"].ge(trips["inicio"]).all()
    assert trips["duracion_min"].between(0, 180).all()
    assert trips.duplicated().mean() < 0.001


def test_p3_profile_dataset_has_valid_ranges_and_unique_stations():
    profile = read_parquet_or_skip(CLUSTERING_FEATURES_FILE)
    required = {
        "estacion_id",
        "dias_activos",
        "salidas",
        "llegadas",
        "viajes_dia",
        "balance",
        "duracion_mediana",
        *P3_PROPORTION_COLUMNS,
    }

    assert not required - set(profile.columns)
    assert not profile[list(required)].isna().any().any()
    assert not profile["estacion_id"].duplicated().any()
    assert profile["dias_activos"].gt(0).all()
    assert profile["viajes_dia"].gt(0).all()
    assert profile["balance"].between(-1, 1).all()
    assert profile[P3_PROPORTION_COLUMNS].apply(lambda values: values.between(0, 1)).all().all()
    assert np.isfinite(profile[list(required - {"estacion_id"})].to_numpy()).all()


def tracked_files() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return [PROJECT_ROOT / name.decode() for name in output.split(b"\0") if name]


def test_sensitive_local_files_are_ignored_and_not_tracked():
    ignore_rules = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in ignore_rules
    assert "*.pem" in ignore_rules
    assert "*.key" in ignore_rules

    forbidden_names = {".env", "credentials.json", "secrets.json"}
    forbidden_suffixes = {".pem", ".key", ".p12", ".pfx"}
    exposed = [
        path.relative_to(PROJECT_ROOT)
        for path in tracked_files()
        if path.name in forbidden_names or path.suffix.lower() in forbidden_suffixes
    ]
    assert not exposed, f"Archivos sensibles versionados: {exposed}"


def test_tracked_text_files_do_not_contain_hardcoded_credentials():
    assignment = re.compile(
        r"(?i)\b(?:api[_-]?key|secret(?:[_-]?key)?|password|passwd|access[_-]?token)"
        r"\b\s*[:=]\s*['\"]([^'\"]{8,})['\"]"
    )
    private_key = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
    findings = []

    for path in tracked_files():
        if path.suffix.lower() not in {".py", ".md", ".toml", ".cfg", ".yaml", ".yml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if assignment.search(text) or private_key.search(text):
            findings.append(str(path.relative_to(PROJECT_ROOT)))

    assert not findings, f"Posibles credenciales embebidas: {findings}"
