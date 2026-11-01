"""Validate the project data with DeepChecks: integrity of every dataset and train/test drift.

There is one command per group of data (raw, p1, p2, p3) and one that runs them all. Each
command saves its HTML reports in reports/deepchecks/<group>/ (ignored by Git because they
show individual trips) and a summary with the result of each condition in
references/validacion/<group>.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from deepchecks.core import CheckFailure, SuiteResult
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import (
    DataDuplicates,
    FeatureDrift,
    IsSingleValue,
    LabelDrift,
    MixedDataTypes,
    MixedNulls,
    MultivariateDrift,
    PercentOfNulls,
)
from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici_ml.config import (
    CLUSTERING_DRIFT_CUTOFF,
    CLUSTERING_FEATURES_FILE,
    CLUSTERING_MANIFEST_FILE,
    CLUSTERING_MAX_DUPLICATE_RATIO,
    CLUSTERING_STATIONS_CLEAN_FILE,
    CLUSTERING_TRIP_FILES,
    CLUSTERING_TRIPS_FILE,
    P1_FEATURES_FILE,
    P1_TRIPS_FILE,
    P2_ALPHA,
    P2_END_DATE,
    P2_START_DATE,
    P2_STATIONS_FILE,
    P2_TRAIN_FRACTION,
    RAW_DATA_DIR,
    VALIDATION_REPORTS_DIR,
    VALIDATION_SUMMARY_DIR,
)
from mibici_ml.dataset import convert_p3_types, read_csv_compatible, read_p3_trips
from mibici_ml.features import (
    build_station_hour_dataset,
    build_station_profile,
    filter_station_profile,
    load_clean_trips,
)
from mibici_ml.modeling.train import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    P3_USAGE_FEATURES,
    chronological_split,
)

app = typer.Typer(no_args_is_help=True)

ALL_ROWS = 10_000_000  # PercentOfNulls revisa 100 mil filas por defecto; así revisa todas
P1_TEST_FRACTION = 0.2  # el mismo valor por defecto de train_p1
TRIP_CATEGORIES = ["origen_id", "destino_id"]
TRIP_DRIFT_CATEGORIES = ["dia_semana", "origen_id", "redondo"]


# ---------------------------------------------------------------------------
# Suites y resumen
# ---------------------------------------------------------------------------


def integrity_suite(max_duplicate_ratio: float = 0.0) -> Suite:
    """Check constant columns, mixed types, nulls, and duplicate rows."""
    return Suite(
        "Integridad de datos",
        IsSingleValue().add_condition_not_single_value(),
        # Con el rango (0, 1) cualquier mezcla de números y texto hace fallar la condición
        MixedDataTypes().add_condition_rare_type_ratio_not_in_range((0, 1)),
        MixedNulls().add_condition_different_nulls_less_equal_to(1),
        PercentOfNulls(n_samples=ALL_ROWS).add_condition_percent_of_nulls_not_greater_than(0),
        DataDuplicates().add_condition_ratio_less_or_equal(max_duplicate_ratio),
    )


def drift_suite(with_label: bool = False) -> Suite:
    """Compare train and test per variable, jointly with a domain classifier, and the label."""
    checks = [
        FeatureDrift().add_condition_drift_score_less_than(),
        MultivariateDrift().add_condition_overall_drift_value_less_than(),
    ]
    if with_label:
        checks.append(LabelDrift().add_condition_drift_score_less_than())
    return Suite("Drift entre train y test", *checks)


def make_dataset(
    frame: pd.DataFrame,
    cat_features: list[str],
    index: str | None = None,
    label: str | None = None,
    label_type: str | None = None,
    datetime: str | None = None,
) -> Dataset:
    """Wrap a table telling DeepChecks the role of each special column."""
    return Dataset(
        frame,
        cat_features=cat_features,
        index_name=index,
        label=label,
        label_type=label_type,
        datetime_name=datetime,
    )


def drift_datasets(
    train: pd.DataFrame, test: pd.DataFrame, cat_features: list[str], **roles
) -> tuple[Dataset, Dataset]:
    """Wrap the train and test tables with the same column roles."""
    return make_dataset(train, cat_features, **roles), make_dataset(test, cat_features, **roles)


def _round(value, digits: int):
    return None if value is None else round(value, digits)


def _main_value(check) -> object:
    """Return the numbers behind each condition so they can be documented."""
    value = check.value
    if isinstance(check.check, DataDuplicates):
        return _round(value, 6)
    if isinstance(check.check, PercentOfNulls):
        return value["Percent of nulls in sample"].round(6).to_dict()
    if isinstance(check.check, FeatureDrift):
        return {column: _round(item["Drift score"], 4) for column, item in value.items()}
    if isinstance(check.check, LabelDrift):
        return _round(value["Drift score"], 4)
    if isinstance(check.check, MultivariateDrift):
        importance = pd.Series(value["domain_classifier_feature_importance"])
        return {
            "drift": _round(value["domain_classifier_drift_score"], 4),
            "auc": _round(value["domain_classifier_auc"], 4),
            "variables_mas_importantes": importance.nlargest(3).round(3).to_dict(),
        }
    return None


def summarize(result: SuiteResult) -> dict:
    """Keep whether each DeepChecks condition passed, its details, and its main numbers."""
    checks = {}
    for check in result.results:
        if isinstance(check, CheckFailure):
            checks[check.get_header()] = {"aprobado": False, "detalle": [str(check.exception)]}
            continue
        conditions = check.conditions_results
        checks[check.get_header()] = {
            "aprobado": all(condition.is_pass() for condition in conditions),
            "detalle": [condition.details for condition in conditions],
            "valor": _main_value(check),
        }
    return {"aprobado": all(item["aprobado"] for item in checks.values()), "chequeos": checks}


def run_suite(suite: Suite, datasets: tuple[Dataset, ...], report_path: Path) -> dict:
    """Run a suite, save its HTML report, and return its summary."""
    result = suite.run(*datasets)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # connected=True carga plotly desde internet y deja el reporte en unos cientos de KB
    result.save_as_html(str(report_path), connected=True)
    summary = summarize(result)
    status = "aprobado" if summary["aprobado"] else "con hallazgos"
    logger.info(f"{report_path.parent.name}/{report_path.stem}: {status}")
    return summary


def save_summary(group: str, results: dict, raw_dir: Path, **details) -> None:
    """Write the summary of a group next to the source data version it came from."""
    manifest_path = raw_dir / "manifest.json"
    source_version = None
    if manifest_path.exists():
        source_version = json.loads(manifest_path.read_text(encoding="utf-8"))["data_version"]
    summary = {"datos": group, "version_fuente": source_version, **details, "reportes": results}
    VALIDATION_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = VALIDATION_SUMMARY_DIR / f"{group}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Resumen en {path}")


def _require(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"No existen {missing}. Corre primero dataset prepare y features.")


# ---------------------------------------------------------------------------
# Viajes crudos (fuente común de P1, P2 y P3)
# ---------------------------------------------------------------------------


def load_raw_p3_trips(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Read the trips with their types, without personal columns and before any cleaning."""
    months = [read_p3_trips(raw_dir / name)[0] for name in CLUSTERING_TRIP_FILES]
    return convert_p3_types(pd.concat(months, ignore_index=True))


def trips_dataset(trips: pd.DataFrame) -> Dataset:
    """Wrap raw or clean trips; the trip id, when present, is used as index."""
    index = "viaje_id" if "viaje_id" in trips.columns else None
    return make_dataset(trips, TRIP_CATEGORIES, index=index)


# ---------------------------------------------------------------------------
# P1 y P2
# ---------------------------------------------------------------------------


def p1_split(
    features: pd.DataFrame, test_fraction: float = P1_TEST_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split the P1 features at the same timestamp cutoff used by train_p1."""
    timestamps = sorted(features["timestamp"].unique())
    cutoff = pd.Timestamp(timestamps[int(len(timestamps) * (1 - test_fraction))])
    before = features["timestamp"] < cutoff
    return features[before], features[~before], cutoff


def build_p2_frame(trips: pd.DataFrame, raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Build the P2 station-hour panel the same way train_p2 does before splitting it."""
    stations = read_csv_compatible(raw_dir / P2_STATIONS_FILE)
    frame = build_station_hour_dataset(trips, stations, alpha=P2_ALPHA)
    hours = frame["hour"]
    in_range = hours.ge(pd.Timestamp(P2_START_DATE)) & hours.lt(pd.Timestamp(P2_END_DATE))
    return frame[in_range].copy()


def p2_risk_label(frame: pd.DataFrame) -> pd.Series:
    """Join the two P2 labels in one category so DeepChecks can measure their drift."""
    bikes = frame["risk_no_bikes"].eq(1)
    docks = frame["risk_no_docks"].eq(1)
    label = np.select(
        [bikes & docks, bikes, docks], ["ambos", "sin_bicis", "sin_anclajes"], "sin_riesgo"
    )
    return pd.Series(label, index=frame.index, name="riesgo")


# ---------------------------------------------------------------------------
# P3
# ---------------------------------------------------------------------------


def profile_dataset(profile: pd.DataFrame) -> Dataset:
    """Wrap the station profile without the station name, which is free text."""
    return make_dataset(profile.drop(columns="nombre"), ["zona"], index="estacion_id")


def split_by_period(
    trips: pd.DataFrame, cutoff: str = CLUSTERING_DRIFT_CUTOFF
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split trips by start date: train before the cutoff and test from the cutoff on."""
    before = trips["inicio"] < pd.Timestamp(cutoff)
    return trips[before], trips[~before]


def trip_drift_table(trips: pd.DataFrame) -> pd.DataFrame:
    """Describe each trip with the variables that feed the station profile."""
    return pd.DataFrame(
        {
            "hora": trips["inicio"].dt.hour,
            "dia_semana": trips["inicio"].dt.dayofweek,
            "duracion_min": trips["duracion_min"],
            "origen_id": trips["origen_id"],
            "redondo": (trips["origen_id"] == trips["destino_id"]).astype(int),
        }
    )


def profile_by_period(
    trips: pd.DataFrame, stations: pd.DataFrame, cutoff: str = CLUSTERING_DRIFT_CUTOFF
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the station profile of each period, keeping the stations present in both.

    Stations opened during the test period would look like drift even if no station changed
    its behavior, so only the common stations are compared.
    """
    train, test = (
        filter_station_profile(build_station_profile(part), stations)[0]
        for part in split_by_period(trips, cutoff)
    )
    common = set(train["estacion_id"]) & set(test["estacion_id"])
    return train[train["estacion_id"].isin(common)], test[test["estacion_id"].isin(common)]


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------


@app.callback()
def main() -> None:
    """Validate the project data with DeepChecks."""
    # En Windows 11 ya no existe wmic y joblib imprime un error al contar los núcleos físicos;
    # con un límite menor a los núcleos lógicos ya no los cuenta
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))


@app.command("raw")
def validate_raw(raw_dir: Path = RAW_DATA_DIR) -> None:
    """Check the 13 monthly trip files that feed P1, P2, and P3."""
    reports = VALIDATION_REPORTS_DIR / "raw"
    trips = load_raw_p3_trips(raw_dir)
    results = {
        "integridad_viajes": run_suite(
            integrity_suite(), (trips_dataset(trips),), reports / "integridad_viajes.html"
        )
    }
    save_summary("raw", results, raw_dir, viajes=len(trips))


@app.command("p1")
def validate_p1(raw_dir: Path = RAW_DATA_DIR) -> None:
    """Check the P1 trips and hourly features, and the drift of its chronological split."""
    _require(P1_TRIPS_FILE, P1_FEATURES_FILE)
    reports = VALIDATION_REPORTS_DIR / "p1"
    trips = pd.read_parquet(P1_TRIPS_FILE)
    features = pd.read_parquet(P1_FEATURES_FILE)
    train, test, cutoff = p1_split(features)
    columns = [*FEATURE_COLUMNS, "target_departures"]
    label = {"label": "target_departures", "label_type": "regression"}
    results = {
        "integridad_viajes": run_suite(
            integrity_suite(),
            (make_dataset(trips, ["Origen_Id", "Destino_Id"], index="Viaje_Id"),),
            reports / "integridad_viajes.html",
        ),
        "integridad_features": run_suite(
            integrity_suite(),
            (make_dataset(features, ["station_id"], datetime="timestamp", **label),),
            reports / "integridad_features.html",
        ),
        "drift": run_suite(
            drift_suite(with_label=True),
            drift_datasets(train[columns], test[columns], ["station_id"], **label),
            reports / "drift.html",
        ),
    }
    save_summary(
        "p1",
        results,
        raw_dir,
        corte_drift=cutoff.isoformat(),
        filas_train=len(train),
        filas_test=len(test),
    )


@app.command("p2")
def validate_p2(raw_dir: Path = RAW_DATA_DIR) -> None:
    """Check the P2 trips and station-hour panel, and the drift of its chronological split."""
    reports = VALIDATION_REPORTS_DIR / "p2"
    trips, _ = load_clean_trips(raw_dir)
    frame = build_p2_frame(trips, raw_dir)
    frame["riesgo"] = p2_risk_label(frame)
    train, test, cutoff = chronological_split(frame, P2_TRAIN_FRACTION)
    columns = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, "riesgo"]
    label = {"label": "riesgo", "label_type": "multiclass"}
    results = {
        "integridad_viajes": run_suite(
            integrity_suite(),
            (make_dataset(trips, ["Origen_Id", "Destino_Id"]),),
            reports / "integridad_viajes.html",
        ),
        "integridad_panel": run_suite(
            integrity_suite(),
            (make_dataset(frame, CATEGORICAL_FEATURES, datetime="hour"),),
            reports / "integridad_panel.html",
        ),
        "drift": run_suite(
            drift_suite(with_label=True),
            drift_datasets(train[columns], test[columns], CATEGORICAL_FEATURES, **label),
            reports / "drift.html",
        ),
    }
    save_summary(
        "p2",
        results,
        raw_dir,
        corte_drift=cutoff.isoformat(),
        filas_train=len(train),
        filas_test=len(test),
        proporcion_riesgo=frame["riesgo"].value_counts(normalize=True).round(4).to_dict(),
    )


@app.command("p3")
def validate_p3(raw_dir: Path = RAW_DATA_DIR) -> None:
    """Check the clean P3 trips and station profile, and the drift between periods."""
    _require(CLUSTERING_TRIPS_FILE, CLUSTERING_STATIONS_CLEAN_FILE, CLUSTERING_FEATURES_FILE)
    reports = VALIDATION_REPORTS_DIR / "p3"
    trips = pd.read_parquet(CLUSTERING_TRIPS_FILE)
    stations = pd.read_parquet(CLUSTERING_STATIONS_CLEAN_FILE)
    profile = pd.read_parquet(CLUSTERING_FEATURES_FILE)
    trip_train, trip_test = (trip_drift_table(part) for part in split_by_period(trips))
    profile_train, profile_test = (
        part[["estacion_id", *P3_USAGE_FEATURES]] for part in profile_by_period(trips, stations)
    )
    results = {
        "integridad_viajes": run_suite(
            integrity_suite(CLUSTERING_MAX_DUPLICATE_RATIO),
            (trips_dataset(trips),),
            reports / "integridad_viajes.html",
        ),
        "integridad_perfil": run_suite(
            integrity_suite(), (profile_dataset(profile),), reports / "integridad_perfil.html"
        ),
        "drift_viajes": run_suite(
            drift_suite(),
            drift_datasets(trip_train, trip_test, TRIP_DRIFT_CATEGORIES),
            reports / "drift_viajes.html",
        ),
        "drift_perfil": run_suite(
            drift_suite(),
            drift_datasets(profile_train, profile_test, [], index="estacion_id"),
            reports / "drift_perfil.html",
        ),
    }
    manifest = json.loads(CLUSTERING_MANIFEST_FILE.read_text(encoding="utf-8"))
    save_summary(
        "p3",
        results,
        raw_dir,
        version_datos_p3=manifest["version_datos"],
        corte_drift=CLUSTERING_DRIFT_CUTOFF,
        viajes_train=len(trip_train),
        viajes_test=len(trip_test),
        estaciones_drift_perfil=len(profile_train),
    )


@app.command("all")
def validate_all(raw_dir: Path = RAW_DATA_DIR) -> None:
    """Run the validations of the raw trips and of the P1, P2, and P3 data."""
    for validate in (validate_raw, validate_p1, validate_p2, validate_p3):
        validate(raw_dir)


if __name__ == "__main__":
    app()
