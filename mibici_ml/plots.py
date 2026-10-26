"""Plotting functions shared by the three modeling pipelines."""

import math
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

BACKGROUND = "#fcfcfb"
TEXT = "#0b0b0b"
SECONDARY_TEXT = "#52514e"
BORDER = "#e1e0dc"
GROUP_COLOR = "#2a78d6"
OTHER_COLOR = "#d6d5cf"
DIVERGING_MAP = LinearSegmentedColormap.from_list("rojo_azul", ["#e34948", "#f0efec", "#2a78d6"])
VARIABLE_NAMES = {
    "viajes_dia": "Viajes por día",
    "balance": "Balance llegadas-salidas",
    "prop_manana": "Salidas 6-10 h",
    "prop_mediodia": "Salidas 10-16 h",
    "prop_tarde": "Salidas 16-20 h",
    "prop_noche": "Salidas 20-6 h",
    "prop_fin_semana": "Salidas en fin de semana",
    "prop_redondo": "Viajes redondos",
    "duracion_mediana": "Duración mediana",
}
MAP_REFERENCES = {
    "Centro GDL": (20.6767, -103.3475),
    "Zapopan": (20.7206, -103.3916),
    "Tlaquepaque": (20.6408, -103.3119),
}


def save_regression_scatter(
    actual: pd.Series, predictions: np.ndarray, model_name: str, output_path: Path
) -> None:
    """Save the P1 observed-versus-predicted sample plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_size = min(5000, len(actual))
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(
        actual.iloc[:sample_size].to_numpy(),
        predictions[:sample_size],
        alpha=0.25,
        s=8,
    )
    axis.set_xlabel("Actual departures")
    axis.set_ylabel("Predicted departures")
    axis.set_title(f"P1: {model_name}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_confusion_matrices(
    y_true: pd.DataFrame, predictions: np.ndarray, targets: list[str], output_path: Path
) -> None:
    """Save the P2 confusion matrices."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4))
    axes = np.atleast_1d(axes)
    for index, target in enumerate(targets):
        ConfusionMatrixDisplay.from_predictions(
            y_true[target], predictions[:, index], ax=axes[index], colorbar=False
        )
        axes[index].set_title(target)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_precision_recall_curves(
    y_true: pd.DataFrame, probabilities: list[np.ndarray], targets: list[str], output_path: Path
) -> None:
    """Save one precision-recall curve per P2 target."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4))
    axes = np.atleast_1d(axes)
    for index, target in enumerate(targets):
        PrecisionRecallDisplay.from_predictions(
            y_true[target], probabilities[index][:, 1], ax=axes[index], name=target
        )
        axes[index].set_title(target)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_cluster_map(assignments: pd.DataFrame) -> Figure:
    """Plot station locations in one panel per P3 cluster."""
    groups = sorted(assignments["grupo"].unique())
    columns = 2 if len(groups) in (2, 4) else 3
    rows = math.ceil(len(groups) / columns)
    figure = Figure(
        figsize=(4.4 * columns, 4.2 * rows), dpi=150, facecolor=BACKGROUND, layout="constrained"
    )
    axes = np.atleast_1d(figure.subplots(rows, columns)).ravel()
    aspect = 1 / math.cos(math.radians(assignments["latitud"].mean()))
    for axis, group in zip(axes, groups):
        in_group = assignments["grupo"] == group
        others, selected = assignments[~in_group], assignments[in_group]
        axis.scatter(others["longitud"], others["latitud"], s=7, color=OTHER_COLOR, linewidths=0)
        axis.scatter(
            selected["longitud"],
            selected["latitud"],
            s=20,
            color=GROUP_COLOR,
            edgecolors=BACKGROUND,
            linewidths=0.6,
        )
        for name, (latitude, longitude) in MAP_REFERENCES.items():
            axis.plot(longitude, latitude, marker="+", color=TEXT, markersize=7)
            axis.annotate(
                name,
                (longitude, latitude),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=SECONDARY_TEXT,
            )
        axis.set_title(f"Grupo {group}: {len(selected)} estaciones", loc="left", color=TEXT)
        axis.set_aspect(aspect)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_facecolor(BACKGROUND)
        for border in axis.spines.values():
            border.set_color(BORDER)
    for axis in axes[len(groups) :]:
        axis.set_visible(False)
    figure.suptitle("Ubicación de cada grupo de estaciones", color=TEXT, fontsize=13)
    return figure


def plot_cluster_profiles(x_eval: np.ndarray, groups: np.ndarray, variables: list[str]) -> Figure:
    """Plot standardized mean feature values for each P3 cluster."""
    means = pd.DataFrame(x_eval, columns=variables).groupby(groups).mean()
    sizes = pd.Series(groups).value_counts()
    limit = max(1.0, float(np.abs(means.to_numpy()).max()))
    figure = Figure(
        figsize=(0.9 * len(variables) + 3, 0.55 * len(means) + 2.6),
        dpi=150,
        facecolor=BACKGROUND,
        layout="constrained",
    )
    axis = figure.subplots()
    image = axis.imshow(
        means.to_numpy(), cmap=DIVERGING_MAP, vmin=-limit, vmax=limit, aspect="auto"
    )
    for (row, column), value in np.ndenumerate(means.to_numpy()):
        axis.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=9)
    axis.set_xticks(
        range(len(variables)),
        [VARIABLE_NAMES.get(variable, variable) for variable in variables],
        rotation=35,
        ha="right",
        fontsize=9,
    )
    axis.set_yticks(
        range(len(means)), [f"Grupo {group} ({sizes[group]})" for group in means.index], fontsize=9
    )
    axis.tick_params(length=0, colors=TEXT)
    for border in axis.spines.values():
        border.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.85)
    colorbar.set_label("Desviaciones estándar respecto al promedio", color=SECONDARY_TEXT)
    colorbar.outline.set_visible(False)
    axis.set_title("Perfil de uso de cada grupo", loc="left", color=TEXT, fontsize=12)
    return figure
