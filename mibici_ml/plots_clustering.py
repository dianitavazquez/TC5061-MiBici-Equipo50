"""Gráficas del clustering de estaciones (P3).

train_clustering.py las usa para registrar cada run en MLflow y guardarlas en
reports/figures. Se usa matplotlib.figure.Figure en lugar de pyplot para no acumular
figuras abiertas cuando se entrenan varios modelos seguidos.
"""

import math

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

FONDO = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
BORDE = "#e1e0dc"
COLOR_GRUPO = "#2a78d6"
COLOR_OTRAS = "#d6d5cf"
# Divergente: rojo por debajo del promedio, gris en el promedio y azul por encima
MAPA_DIVERGENTE = LinearSegmentedColormap.from_list("rojo_azul", ["#e34948", "#f0efec", "#2a78d6"])

NOMBRES_VARIABLES = {
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

# Puntos de referencia para ubicarse en el mapa (coordenadas aproximadas)
REFERENCIAS = {
    "Centro GDL": (20.6767, -103.3475),
    "Zapopan": (20.7206, -103.3916),
    "Tlaquepaque": (20.6408, -103.3119),
}


def graficar_mapa(asignacion):
    """Mapa de las estaciones con un panel por grupo.

    Cada panel resalta las estaciones de un grupo en azul y deja las demás en gris, así se
    ve dónde se ubica cada grupo sin tener que distinguir muchos colores en un solo mapa.

    asignacion: DataFrame con las columnas latitud, longitud y grupo.
    """
    grupos = sorted(asignacion["grupo"].unique())
    n_grupos = len(grupos)
    columnas = 2 if n_grupos in (2, 4) else 3
    filas = math.ceil(n_grupos / columnas)
    fig = Figure(
        figsize=(4.4 * columnas, 4.2 * filas), dpi=150, facecolor=FONDO, layout="constrained"
    )
    ejes = np.atleast_1d(fig.subplots(filas, columnas)).ravel()
    # Corrige la proporción para que un grado de longitud no se vea igual que uno de latitud
    aspecto = 1 / math.cos(math.radians(asignacion["latitud"].mean()))

    for eje, grupo in zip(ejes, grupos):
        es_grupo = asignacion["grupo"] == grupo
        otras, del_grupo = asignacion[~es_grupo], asignacion[es_grupo]
        eje.scatter(otras["longitud"], otras["latitud"], s=7, color=COLOR_OTRAS, linewidths=0)
        eje.scatter(
            del_grupo["longitud"],
            del_grupo["latitud"],
            s=20,
            color=COLOR_GRUPO,
            edgecolors=FONDO,
            linewidths=0.6,
        )
        for nombre, (latitud, longitud) in REFERENCIAS.items():
            eje.plot(longitud, latitud, marker="+", color=TEXTO, markersize=7)
            eje.annotate(
                nombre,
                (longitud, latitud),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=TEXTO_SECUNDARIO,
            )
        eje.set_title(f"Grupo {grupo}: {len(del_grupo)} estaciones", loc="left", color=TEXTO)
        eje.set_aspect(aspecto)
        eje.set_xticks([])
        eje.set_yticks([])
        eje.set_facecolor(FONDO)
        for borde in eje.spines.values():
            borde.set_color(BORDE)

    for eje in ejes[n_grupos:]:
        eje.set_visible(False)
    fig.suptitle("Ubicación de cada grupo de estaciones", color=TEXTO, fontsize=13)
    return fig


def graficar_perfiles(x_eval, grupos, variables):
    """Heatmap con el promedio estandarizado de cada variable por grupo.

    0 es el promedio de todas las estaciones. Un valor de 1 significa que el grupo está una
    desviación estándar por encima del promedio (azul) y -1, una por debajo (rojo).

    x_eval: matriz estandarizada (una fila por estación) en el orden de variables.
    grupos: grupo de cada estación.
    variables: nombres de las columnas de x_eval.
    """
    promedios = pd.DataFrame(x_eval, columns=variables).groupby(grupos).mean()
    tamanos = pd.Series(grupos).value_counts()
    limite = max(1.0, float(np.abs(promedios.to_numpy()).max()))

    fig = Figure(
        figsize=(0.9 * len(variables) + 3, 0.55 * len(promedios) + 2.6),
        dpi=150,
        facecolor=FONDO,
        layout="constrained",
    )
    eje = fig.subplots()
    imagen = eje.imshow(
        promedios.to_numpy(), cmap=MAPA_DIVERGENTE, vmin=-limite, vmax=limite, aspect="auto"
    )
    for (fila, columna), valor in np.ndenumerate(promedios.to_numpy()):
        eje.text(columna, fila, f"{valor:.1f}", ha="center", va="center", fontsize=9)

    etiquetas_x = [NOMBRES_VARIABLES.get(v, v) for v in variables]
    etiquetas_y = [f"Grupo {g} ({tamanos[g]})" for g in promedios.index]
    eje.set_xticks(range(len(variables)), etiquetas_x, rotation=35, ha="right", fontsize=9)
    eje.set_yticks(range(len(promedios)), etiquetas_y, fontsize=9)
    eje.tick_params(length=0, colors=TEXTO)
    for borde in eje.spines.values():
        borde.set_visible(False)
    barra = fig.colorbar(imagen, ax=eje, shrink=0.85)
    barra.set_label("Desviaciones estándar respecto al promedio", color=TEXTO_SECUNDARIO)
    barra.outline.set_visible(False)
    eje.set_title("Perfil de uso de cada grupo", loc="left", color=TEXTO, fontsize=12)
    return fig
