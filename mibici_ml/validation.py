"""Validate the P3 data with DeepChecks: integrity of trips and profile, and train/test drift.

The HTML reports are saved in reports/deepchecks/ (ignored by Git because they show individual
trips) and a summary with the result of each condition in references/validacion_clustering.json.
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
    MixedDataTypes,
    MixedNulls,
    MultivariateDrift,
    PercentOfNulls,
)
from loguru import logger
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
    CLUSTERING_VALIDATION_DIR,
    CLUSTERING_VALIDATION_FILE,
    RAW_DATA_DIR,
)
from mibici_ml.dataset import convert_p3_types, read_p3_trips
from mibici_ml.features import build_station_profile, filter_station_profile
from mibici_ml.modeling.train import P3_USAGE_FEATURES

app = typer.Typer(no_args_is_help=True)

ALL_ROWS = 10_000_000  # PercentOfNulls revisa 100 mil filas por defecto; así revisa todas
TRIP_CATEGORIES = ["origen_id", "destino_id"]
TRIP_DRIFT_CATEGORIES = ["dia_semana", "origen_id", "redondo"]


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


def drift_suite() -> Suite:
    """Compare train and test per variable and jointly with a domain classifier."""
    return Suite(
        "Drift entre train y test",
        FeatureDrift().add_condition_drift_score_less_than(),
        MultivariateDrift().add_condition_overall_drift_value_less_than(),
    )


def load_raw_p3_trips(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Read the P3 trips with their types, before applying the cleaning rules."""
    months = [read_p3_trips(raw_dir / name)[0] for name in CLUSTERING_TRIP_FILES]
    return convert_p3_types(pd.concat(months, ignore_index=True))


def trips_dataset(trips: pd.DataFrame) -> Dataset:
    """Wrap raw or clean trips; the trip id, when present, is used as index."""
    index = "viaje_id" if "viaje_id" in trips.columns else None
    return Dataset(trips, cat_features=TRIP_CATEGORIES, index_name=index)


def profile_dataset(profile: pd.DataFrame) -> Dataset:
    """Wrap the station profile without the station name, which is free text."""
    return Dataset(profile.drop(columns="nombre"), cat_features=["zona"], index_name="estacion_id")


def drift_datasets(
    train: pd.DataFrame, test: pd.DataFrame, cat_features: list[str], index: str | None = None
) -> tuple[Dataset, Dataset]:
    """Wrap the train and test tables with the same column roles."""
    return (
        Dataset(train, cat_features=cat_features, index_name=index),
        Dataset(test, cat_features=cat_features, index_name=index),
    )


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


def _main_value(check) -> object:
    """Return the number behind each condition so it can be documented."""
    value = check.value
    if isinstance(check.check, DataDuplicates):
        return round(value, 6)
    if isinstance(check.check, PercentOfNulls):
        return value["Percent of nulls in sample"].round(6).to_dict()
    if isinstance(check.check, FeatureDrift):
        return {column: round(item["Drift score"], 4) for column, item in value.items()}
    if isinstance(check.check, MultivariateDrift):
        return round(value["domain_classifier_drift_score"], 4)
    return None


def summarize(result: SuiteResult) -> dict:
    """Keep whether each DeepChecks condition passed, its details, and its main number."""
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
    # connected=True carga plotly desde internet y deja el reporte en unos cientos de KB
    result.save_as_html(str(report_path), connected=True)
    summary = summarize(result)
    status = "aprobado" if summary["aprobado"] else "con hallazgos"
    logger.info(f"{report_path.stem}: {status}")
    return summary


@app.callback()
def main() -> None:
    """Data validations with DeepChecks (one command per question)."""


@app.command("p3")
def validate_p3(
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = CLUSTERING_VALIDATION_DIR,
    summary_path: Path = CLUSTERING_VALIDATION_FILE,
) -> None:
    """Run the DeepChecks suites on the P3 data and save the reports and summary."""
    for path in [CLUSTERING_TRIPS_FILE, CLUSTERING_STATIONS_CLEAN_FILE, CLUSTERING_FEATURES_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"No existe {path}. Corre primero prepare-p3 y build-p3.")
    # En Windows 11 ya no existe wmic y joblib imprime un error al contar los núcleos físicos;
    # con un límite menor a los núcleos lógicos ya no los cuenta
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))
    output_dir.mkdir(parents=True, exist_ok=True)
    trips = pd.read_parquet(CLUSTERING_TRIPS_FILE)
    stations = pd.read_parquet(CLUSTERING_STATIONS_CLEAN_FILE)
    profile = pd.read_parquet(CLUSTERING_FEATURES_FILE)
    trip_train, trip_test = (trip_drift_table(part) for part in split_by_period(trips))
    profile_train, profile_test = (
        part[["estacion_id", *P3_USAGE_FEATURES]] for part in profile_by_period(trips, stations)
    )

    results = {
        "integridad_viajes_crudos": run_suite(
            integrity_suite(),
            (trips_dataset(load_raw_p3_trips(raw_dir)),),
            output_dir / "integridad_viajes_crudos.html",
        ),
        "integridad_viajes_limpios": run_suite(
            integrity_suite(CLUSTERING_MAX_DUPLICATE_RATIO),
            (trips_dataset(trips),),
            output_dir / "integridad_viajes_limpios.html",
        ),
        "integridad_perfil": run_suite(
            integrity_suite(),
            (profile_dataset(profile),),
            output_dir / "integridad_perfil.html",
        ),
        "drift_viajes": run_suite(
            drift_suite(),
            drift_datasets(trip_train, trip_test, TRIP_DRIFT_CATEGORIES),
            output_dir / "drift_viajes.html",
        ),
        "drift_perfil": run_suite(
            drift_suite(),
            drift_datasets(profile_train, profile_test, [], "estacion_id"),
            output_dir / "drift_perfil.html",
        ),
    }

    manifest = json.loads(CLUSTERING_MANIFEST_FILE.read_text(encoding="utf-8"))
    summary = {
        "pregunta": "P3",
        "version_datos": manifest["version_datos"],
        "corte_drift": CLUSTERING_DRIFT_CUTOFF,
        "viajes_train": len(trip_train),
        "viajes_test": len(trip_test),
        "estaciones_drift_perfil": len(profile_train),
        "reportes": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Reportes en {output_dir} y resumen en {summary_path}")


if __name__ == "__main__":
    app()
