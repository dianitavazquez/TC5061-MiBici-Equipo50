"""MLflow Project entry-point dispatch for the three modeling questions."""

from __future__ import annotations

import json
from typing import Any

import typer

from mibici_ml.config import (
    CLUSTERING_K,
    CLUSTERING_N_INIT,
    P1_L2_REGULARIZATION,
    P1_LEARNING_RATE,
    P1_MAX_ITER,
    P1_MAX_LEAF_NODES,
    P1_TEST_FRACTION,
    P2_MAX_ITER,
)
from mibici_ml.features import build_p1_command, build_p3_command
from mibici_ml.modeling.train import search_p2, train_p1, train_p2, train_p3

app = typer.Typer(no_args_is_help=True)


def parse_hyperparameters(value: str) -> dict[str, Any]:
    """Parse and validate the JSON object passed through ``mlflow run -P``."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise typer.BadParameter(f"hyperparameters debe ser JSON válido: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise typer.BadParameter("hyperparameters debe ser un objeto JSON.")
    return parsed


def validate_hyperparameters(
    question: str, supplied: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any]:
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise typer.BadParameter(
            f"Hiperparámetros no válidos para {question}: {sorted(unknown)}. "
            f"Usa {sorted(defaults)}."
        )
    return {**defaults, **supplied}


@app.command("features")
def build_features(question: str = typer.Option(...)) -> None:
    """Build the persisted features needed by a question."""
    question = question.lower()
    if question == "p1":
        build_p1_command()
    elif question == "p2":
        typer.echo("P2 construye sus variables sin fuga temporal durante el entrenamiento.")
    elif question == "p3":
        build_p3_command()
    else:
        raise typer.BadParameter("question debe ser p1, p2 o p3.")


@app.command("p2-search")
def run_p2_search() -> None:
    """Compare the P2 baseline and three candidates as nested MLflow runs."""
    search_p2()


@app.command("train")
def train(
    question: str = typer.Option(...),
    model: str = typer.Option(...),
    hyperparameters: str = typer.Option("{}"),
) -> None:
    """Train one selected model with question-specific JSON hyperparameters."""
    question = question.lower()
    model = model.lower()
    supplied = parse_hyperparameters(hyperparameters)

    if question == "p1":
        params = validate_hyperparameters(
            question,
            supplied,
            {
                "test_fraction": P1_TEST_FRACTION,
                "max_iter": P1_MAX_ITER,
                "learning_rate": P1_LEARNING_RATE,
                "max_leaf_nodes": P1_MAX_LEAF_NODES,
                "l2_regularization": P1_L2_REGULARIZATION,
            },
        )
        train_p1(models=(model,), **params)
    elif question == "p2":
        params = validate_hyperparameters(
            question,
            supplied,
            {"c": None, "class_weight": None, "max_iter": P2_MAX_ITER},
        )
        train_p2(models=(model,), model_params=params)
    elif question == "p3":
        params = validate_hyperparameters(
            question, supplied, {"k": CLUSTERING_K, "n_init": CLUSTERING_N_INIT}
        )
        train_p3(models=(model,), ks=(int(params["k"]),), n_init=int(params["n_init"]))
    else:
        raise typer.BadParameter("question debe ser p1, p2 o p3.")


if __name__ == "__main__":
    app()
