"""A/B test of the P2 champion against a challenger on the same current-period data."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import typer

from mibici_ml.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    P2_REGISTERED_MODEL_NAME,
    PROJ_ROOT,
    RANDOM_SEED,
    REFERENCES_DIR,
)
from mibici_ml.modeling.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGETS,
    evaluate,
    git_commit,
)
from mibici_ml.plots import BACKGROUND, BORDER, GROUP_COLOR, SECONDARY_TEXT, TEXT

app = typer.Typer(no_args_is_help=True)

FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
VARIANTS = ("champion", "challenger")

# Octubre de 2026 es el periodo actual del monitoreo; el challenger solo puede entrenar con
# fechas anteriores a AB_EVALUATION_START para que ningún modelo haya visto la evaluación.
AB_EVALUATION_FILE = INTERIM_DATA_DIR / "p2_current_2026_10.parquet"
AB_EVALUATION_START = "2026-10-16"
AB_EVALUATION_END = "2026-11-01"
AB_CHAMPION_URI = str(MODELS_DIR / "serving" / "p2")
AB_CHALLENGER_URI = str(MODELS_DIR / "serving" / "p2_challenger")
AB_BOOTSTRAP_SAMPLES = 1000
AB_CONFIDENCE = 0.95
AB_MAX_RECALL_DROP = 0.05  # pérdida máxima de recall que se tolera en cada riesgo
AB_SUMMARY_DIR = REFERENCES_DIR / "ab_testing"
AB_FIGURE_DIR = FIGURES_DIR / "ab_testing"
AB_OUTPUT_NAME = "p2_ab_2026_10"

CHAMPION_COLOR = GROUP_COLOR
CHALLENGER_COLOR = "#eb6834"
NEUTRAL_COLOR = "#a3a29c"
TARGET_LABELS = {"risk_no_bikes": "Sin bicis", "risk_no_docks": "Sin anclajes"}
METRIC_LABELS = {"f1": "F1", "precision": "Precisión", "recall": "Recall"}


@app.callback()
def main():
    """Compare two P2 model versions on the same evaluation set."""


def load_evaluation_frame(
    path: Path = AB_EVALUATION_FILE,
    start: str = AB_EVALUATION_START,
    end: str = AB_EVALUATION_END,
) -> pd.DataFrame:
    """Load the current-period station-hours that both models are evaluated on."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Arma primero el periodo actual con monitor-p2."
        )
    start_timestamp, end_timestamp = pd.Timestamp(start), pd.Timestamp(end)
    if start_timestamp >= end_timestamp:
        raise ValueError("start debe ser anterior a end.")
    frame = pd.read_parquet(path, columns=["hour", *FEATURES, *TARGETS])
    window = frame["hour"].ge(start_timestamp) & frame["hour"].lt(end_timestamp)
    frame = frame[window].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No hay filas entre {start} y {end}.")
    without_positives = [target for target in TARGETS if frame[target].sum() == 0]
    if without_positives:
        raise ValueError(f"La ventana no tiene casos positivos de {without_positives}.")
    return frame


def describe_model_source(uri: str) -> dict:
    """Return a portable reference to the model and its registry metadata when available."""
    path = Path(uri)
    if not path.exists():
        return {"uri": uri}
    try:
        shown = path.resolve().relative_to(PROJ_ROOT).as_posix()
    except ValueError:
        shown = path.name
    source = {"uri": shown}
    info_file = path / "model-info.json"
    if info_file.exists():
        info = json.loads(info_file.read_text(encoding="utf-8"))
        source.update(
            {key: info[key] for key in ("registered_model", "version", "run_id") if key in info}
        )
    return source


def load_model(uri: str, tracking_uri: str = MLFLOW_TRACKING_URI):
    """Load a local model copy or a registry URI such as models:/name/version."""
    if not Path(uri).exists():
        mlflow.set_tracking_uri(tracking_uri)
    return mlflow.sklearn.load_model(str(uri))


def score_model(model, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    """Predict the evaluation set and compute the same metrics used to pick the champion."""
    predictions = model.predict(frame[FEATURES])
    probabilities = model.predict_proba(frame[FEATURES])
    metrics = evaluate(frame[TARGETS], predictions, probabilities)
    return predictions, {name: float(value) for name, value in metrics.items()}


def station_confusion_counts(
    stations: np.ndarray, actual: np.ndarray, predicted: np.ndarray
) -> np.ndarray:
    """Count TP, FP and FN per station and target, with shape (stations, targets, 3)."""
    codes, uniques = pd.factorize(stations, sort=True)
    counts = np.zeros((len(uniques), actual.shape[1], 3))
    for target in range(actual.shape[1]):
        alarm = predicted[:, target] == 1
        risk = actual[:, target] == 1
        for kind, mask in enumerate((alarm & risk, alarm & ~risk, ~alarm & risk)):
            counts[:, target, kind] = np.bincount(
                codes, weights=mask.astype(float), minlength=len(uniques)
            )
    return counts


def macro_f1_from_counts(counts: np.ndarray) -> np.ndarray:
    """Average the per-target F1 stored as TP, FP and FN in the last axis (0 if undefined)."""
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    denominator = 2 * tp + fp + fn
    f1 = np.divide(
        2 * tp, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0
    )
    return f1.mean(axis=-1)


def paired_station_bootstrap(
    frame: pd.DataFrame,
    champion_predictions: np.ndarray,
    challenger_predictions: np.ndarray,
    samples: int = AB_BOOTSTRAP_SAMPLES,
    confidence: float = AB_CONFIDENCE,
    seed: int = RANDOM_SEED,
) -> tuple[dict, np.ndarray]:
    """Confidence interval of the macro-F1 difference resampling whole stations.

    Las horas de una misma estación están correlacionadas, así que se remuestrean estaciones
    completas y ambos modelos se evalúan con exactamente los mismos remuestreos.
    """
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("samples debe ser positivo y confidence estar en (0, 1).")
    stations = frame["station_id"].to_numpy()
    actual = frame[TARGETS].to_numpy()
    champion = station_confusion_counts(stations, actual, champion_predictions)
    challenger = station_confusion_counts(stations, actual, challenger_predictions)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(champion), size=(samples, len(champion)))
    differences = macro_f1_from_counts(challenger[indices].sum(axis=1)) - macro_f1_from_counts(
        champion[indices].sum(axis=1)
    )
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(differences, [tail, 100 - tail])
    observed = macro_f1_from_counts(challenger.sum(axis=0)) - macro_f1_from_counts(
        champion.sum(axis=0)
    )
    summary = {
        "unit": "station_id",
        "stations": len(champion),
        "samples": samples,
        "seed": seed,
        "confidence": confidence,
        "observed_difference": float(observed),
        "mean_difference": float(differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_challenger_better": float((differences > 0).mean()),
    }
    return summary, differences


def mcnemar_by_target(
    frame: pd.DataFrame, champion_predictions: np.ndarray, challenger_predictions: np.ndarray
) -> dict[str, dict]:
    """Exact McNemar test per target; informative only because negative hours dominate."""
    results = {}
    for index, target in enumerate(TARGETS):
        actual = frame[target].to_numpy()
        champion_correct = champion_predictions[:, index] == actual
        challenger_correct = challenger_predictions[:, index] == actual
        only_champion = int((champion_correct & ~challenger_correct).sum())
        only_challenger = int((~champion_correct & challenger_correct).sum())
        discordant = only_champion + only_challenger
        p_value = binomtest(only_challenger, discordant, 0.5).pvalue if discordant else 1.0
        results[target] = {
            "only_champion_correct": only_champion,
            "only_challenger_correct": only_challenger,
            "p_value": float(p_value),
        }
    return results


def decide(
    champion_metrics: dict[str, float],
    challenger_metrics: dict[str, float],
    bootstrap: dict,
    max_recall_drop: float = AB_MAX_RECALL_DROP,
) -> dict:
    """Promote the challenger only with a clear macro-F1 gain and no large recall loss."""
    percent = f"{bootstrap['confidence']:.0%}"
    recall_drop = {
        target: champion_metrics[f"{target}_recall"] - challenger_metrics[f"{target}_recall"]
        for target in TARGETS
    }
    clear_gain = bootstrap["ci_low"] > 0
    recall_kept = all(drop <= max_recall_drop for drop in recall_drop.values())
    interval = f"[{bootstrap['ci_low']:+.4f}, {bootstrap['ci_high']:+.4f}]"
    reasons = [
        (
            f"El IC {percent} de la mejora en macro-F1 {interval} queda arriba de 0."
            if clear_gain
            else f"El IC {percent} de la mejora en macro-F1 {interval} incluye valores <= 0."
        )
    ]
    for target, drop in recall_drop.items():
        state = "dentro" if drop <= max_recall_drop else "fuera"
        reasons.append(
            f"Cambio de recall en {target}: {-drop:+.4f}, {state} del límite de "
            f"-{max_recall_drop:.2f}."
        )
    promote = clear_gain and recall_kept
    return {
        "winner": "challenger" if promote else "champion",
        "promote_challenger": promote,
        "rule": (
            f"Promover el challenger solo si el límite inferior del IC {percent} de la "
            f"diferencia de macro-F1 es mayor que 0 y ningún recall baja más de "
            f"{max_recall_drop:.2f}."
        ),
        "recall_drop": recall_drop,
        "reasons": reasons,
    }


def comparison_table(
    champion_metrics: dict[str, float], challenger_metrics: dict[str, float]
) -> pd.DataFrame:
    """Side-by-side metrics with the challenger minus champion difference."""
    table = pd.DataFrame({"champion": champion_metrics, "challenger": challenger_metrics})
    table["difference"] = table["challenger"] - table["champion"]
    rows = ["macro_f1"] + [
        f"{target}_{metric}"
        for target in TARGETS
        for metric in ("f1", "precision", "recall", "average_precision", "roc_auc")
    ]
    table = table.loc[rows]
    table.index.name = "metric"
    return table


def _style_axis(axis) -> None:
    axis.set_facecolor(BACKGROUND)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(BORDER)
    axis.tick_params(colors=SECONDARY_TEXT, labelsize=9)
    axis.grid(axis="x", color=BORDER, linewidth=0.8)
    axis.set_axisbelow(True)


def plot_ab_results(
    table: pd.DataFrame, differences: np.ndarray, bootstrap: dict, title: str, path: Path
) -> Path:
    """Save the per-target metric bars and the bootstrap distribution of the difference."""
    figure = Figure(figsize=(11, 4.4), facecolor=BACKGROUND, layout="constrained")
    bars_axis, bootstrap_axis = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.3, 1]})
    rows = [(target, metric) for target in TARGETS for metric in METRIC_LABELS]
    positions = np.arange(len(rows))
    height = 0.4
    for offset, variant, color in (
        (-0.21, "champion", CHAMPION_COLOR),
        (0.21, "challenger", CHALLENGER_COLOR),
    ):
        values = [table.loc[f"{target}_{metric}", variant] for target, metric in rows]
        bars_axis.barh(
            positions + offset,
            values,
            height=height,
            color=color,
            edgecolor=BACKGROUND,
            linewidth=1,
            label=variant.capitalize(),
        )
    labels = [f"{TARGET_LABELS[target]}: {METRIC_LABELS[metric]}" for target, metric in rows]
    bars_axis.set_yticks(positions, labels)
    bars_axis.invert_yaxis()
    bars_axis.set_xlabel("Valor en la evaluación", color=SECONDARY_TEXT, fontsize=9)
    bars_axis.set_title("Métricas por riesgo", color=TEXT, fontsize=11, loc="left")
    _style_axis(bars_axis)

    low, high = bootstrap["ci_low"], bootstrap["ci_high"]
    bootstrap_axis.hist(
        differences, bins=40, color=NEUTRAL_COLOR, edgecolor=BACKGROUND, linewidth=0.6
    )
    bootstrap_axis.axvspan(low, high, color=BORDER, alpha=0.5, zorder=0)
    for value in (low, high):
        bootstrap_axis.axvline(value, color=SECONDARY_TEXT, linestyle="--", linewidth=1)
    bootstrap_axis.axvline(0, color=TEXT, linewidth=1.2)
    bootstrap_axis.set_xlabel(
        "Diferencia de macro-F1 (challenger - champion)", color=SECONDARY_TEXT, fontsize=9
    )
    bootstrap_axis.set_ylabel("Remuestreos", color=SECONDARY_TEXT, fontsize=9)
    bootstrap_axis.set_title(
        f"Bootstrap por estación ({bootstrap['samples']:,} remuestreos)",
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    bootstrap_axis.text(
        0.02,
        0.97,
        f"IC {bootstrap['confidence']:.0%}: [{low:+.3f}, {high:+.3f}]\n"
        f"P(challenger mejor) = {bootstrap['probability_challenger_better']:.2f}",
        transform=bootstrap_axis.transAxes,
        va="top",
        fontsize=9,
        color=TEXT,
        bbox={"facecolor": BACKGROUND, "edgecolor": BORDER, "boxstyle": "round,pad=0.4"},
        zorder=5,
    )
    bootstrap_axis.set_ylim(top=bootstrap_axis.get_ylim()[1] * 1.3)
    _style_axis(bootstrap_axis)
    figure.suptitle(title, color=TEXT, fontsize=12, x=0.01, ha="left")
    figure.legend(
        *bars_axis.get_legend_handles_labels(),
        loc="outside upper right",
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=TEXT,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, facecolor=BACKGROUND)
    return path


def log_ab_run(
    summary: dict,
    table: pd.DataFrame,
    artifacts: list[Path],
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT,
) -> str:
    """Log the comparison as one MLflow run next to the P2 training runs."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    evaluation, bootstrap = summary["evaluation"], summary["bootstrap"]
    metrics = {
        f"{column}_{metric}": float(value)
        for column in table.columns
        for metric, value in table[column].items()
    }
    metrics.update(
        {
            "bootstrap_mean_difference": bootstrap["mean_difference"],
            "bootstrap_ci_low": bootstrap["ci_low"],
            "bootstrap_ci_high": bootstrap["ci_high"],
            "bootstrap_probability_challenger_better": bootstrap["probability_challenger_better"],
        }
    )
    with mlflow.start_run(run_name=f"p2-ab-test-{summary['output_name']}") as run:
        mlflow.set_tags(
            {
                "team": "50",
                "question": "P2",
                "stage_of_work": "ab_test",
                "registered_model": P2_REGISTERED_MODEL_NAME,
                "winner": summary["decision"]["winner"],
                "git_commit": summary["git_commit"],
                "contains_personal_data": "false",
            }
        )
        mlflow.log_params(
            {
                "champion_uri": summary["champion"]["uri"],
                "challenger_uri": summary["challenger"]["uri"],
                "champion_version": summary["champion"].get("version", "local"),
                "challenger_version": summary["challenger"].get("version", "local"),
                "evaluation_start": evaluation["start"],
                "evaluation_end_exclusive": evaluation["end_exclusive"],
                "evaluation_rows": evaluation["rows"],
                "evaluation_stations": evaluation["stations"],
                "bootstrap_unit": bootstrap["unit"],
                "bootstrap_samples": bootstrap["samples"],
                "bootstrap_confidence": bootstrap["confidence"],
                "seed": bootstrap["seed"],
                "max_recall_drop": summary["decision_threshold"]["max_recall_drop"],
            }
        )
        mlflow.log_metrics(metrics)
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact), artifact_path="ab_test")
        mlflow.log_dict(summary, "ab_test/summary.json")
    return run.info.run_id


def run_ab_test(
    champion_uri: str = AB_CHAMPION_URI,
    challenger_uri: str = AB_CHALLENGER_URI,
    evaluation_path: Path = AB_EVALUATION_FILE,
    start: str = AB_EVALUATION_START,
    end: str = AB_EVALUATION_END,
    samples: int = AB_BOOTSTRAP_SAMPLES,
    confidence: float = AB_CONFIDENCE,
    max_recall_drop: float = AB_MAX_RECALL_DROP,
    seed: int = RANDOM_SEED,
    summary_dir: Path = AB_SUMMARY_DIR,
    figure_dir: Path = AB_FIGURE_DIR,
    output_name: str = AB_OUTPUT_NAME,
    log_to_mlflow: bool = True,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT,
) -> dict:
    """Evaluate champion and challenger on the same rows and decide which one stays."""
    frame = load_evaluation_frame(evaluation_path, start, end)
    logger.info(
        "Evaluando con {:,} filas de {} estaciones entre {} y {}",
        len(frame),
        frame["station_id"].nunique(),
        frame["hour"].min(),
        frame["hour"].max(),
    )
    predictions, metrics, sources = {}, {}, {}
    for variant, uri in zip(VARIANTS, (champion_uri, challenger_uri)):
        logger.info("Cargando {} desde {}", variant, uri)
        model = load_model(str(uri), tracking_uri)
        predictions[variant], metrics[variant] = score_model(model, frame)
        sources[variant] = describe_model_source(str(uri))

    bootstrap, differences = paired_station_bootstrap(
        frame, predictions["champion"], predictions["challenger"], samples, confidence, seed
    )
    decision = decide(metrics["champion"], metrics["challenger"], bootstrap, max_recall_drop)
    table = comparison_table(metrics["champion"], metrics["challenger"])
    summary = {
        "question": "P2",
        "registered_model": P2_REGISTERED_MODEL_NAME,
        "output_name": output_name,
        "git_commit": git_commit(),
        "champion": sources["champion"],
        "challenger": sources["challenger"],
        "evaluation": {
            "file": Path(evaluation_path).name,
            "start": str(pd.Timestamp(start).date()),
            "end_exclusive": str(pd.Timestamp(end).date()),
            "first_hour": frame["hour"].min().isoformat(),
            "last_hour": frame["hour"].max().isoformat(),
            "rows": len(frame),
            "stations": int(frame["station_id"].nunique()),
            "positives": {target: int(frame[target].sum()) for target in TARGETS},
            "positive_rate": {target: float(frame[target].mean()) for target in TARGETS},
        },
        "metrics": {column: table[column].to_dict() for column in table.columns},
        "bootstrap": bootstrap,
        "mcnemar": mcnemar_by_target(frame, predictions["champion"], predictions["challenger"]),
        "decision_threshold": {"max_recall_drop": max_recall_drop},
        "decision": decision,
    }

    summary_dir, figure_dir = Path(summary_dir), Path(figure_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    table_path = summary_dir / f"{output_name}.csv"
    table.round(6).to_csv(table_path)
    title = (
        f"A/B de P2 con {len(frame):,} estación-horas del "
        f"{summary['evaluation']['start']} al {frame['hour'].max():%Y-%m-%d}"
    )
    figure_path = plot_ab_results(
        table, differences, bootstrap, title, figure_dir / f"{output_name}.png"
    )
    summary_path = summary_dir / f"{output_name}.json"
    if log_to_mlflow:
        summary["mlflow_run_id"] = log_ab_run(
            summary, table, [table_path, figure_path], tracking_uri, experiment_name
        )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("\n{}", table.round(4).to_string())
    logger.success("Gana {}: {}", decision["winner"], " ".join(decision["reasons"]))
    return summary


@app.command("run")
def run_command(
    champion: str = typer.Option(AB_CHAMPION_URI, help="Ruta local o URI models:/ del champion"),
    challenger: str = typer.Option(
        AB_CHALLENGER_URI, help="Ruta local o URI models:/ del challenger"
    ),
    evaluation_path: Path = typer.Option(AB_EVALUATION_FILE, help="Parquet del periodo actual"),
    start: str = typer.Option(AB_EVALUATION_START, help="Primer día evaluado (incluido)"),
    end: str = typer.Option(AB_EVALUATION_END, help="Día en que termina la evaluación (excluido)"),
    samples: int = typer.Option(AB_BOOTSTRAP_SAMPLES, help="Remuestreos del bootstrap"),
    summary_dir: Path = typer.Option(AB_SUMMARY_DIR, help="Carpeta del resumen JSON y la tabla"),
    figure_dir: Path = typer.Option(AB_FIGURE_DIR, help="Carpeta de la figura"),
    output_name: str = typer.Option(AB_OUTPUT_NAME, help="Nombre base de las salidas"),
    log_to_mlflow: bool = typer.Option(
        True, "--mlflow/--no-mlflow", help="Registrar la comparación en MLflow"
    ),
) -> None:
    """Run the A/B test of the P2 champion against the challenger."""
    run_ab_test(
        champion_uri=champion,
        challenger_uri=challenger,
        evaluation_path=evaluation_path,
        start=start,
        end=end,
        samples=samples,
        summary_dir=summary_dir,
        figure_dir=figure_dir,
        output_name=output_name,
        log_to_mlflow=log_to_mlflow,
    )


if __name__ == "__main__":
    app()
