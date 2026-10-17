"""Entrena los modelos de clustering de estaciones (P3) y los registra en MLflow.

Cada combinación de modelo y número de grupos (k) es un run del experimento, con sus
parámetros, métricas, artefactos (tablas y figuras), el modelo entrenado y las etiquetas de
equipo, pregunta y commit de Git. Las figuras también se guardan en reports/figures.

Modelos:
    baseline: K-Means solo con viajes por día y proporción de fin de semana.
    candidato: K-Means con todo el perfil de uso de la estación.

Para que las métricas se puedan comparar entre modelos, siempre se calculan en el mismo
espacio: todas las variables de uso, con logaritmo en viajes_dia y estandarizadas.

Uso (desde la raíz del repo):
    python -m mibici_ml.modeling.train_clustering
    python -m mibici_ml.modeling.train_clustering --modelo candidato --k 3 --k 5 --k 6
"""

import json
import subprocess

from loguru import logger
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
import typer

from mibici_ml.config import (
    CLUSTERING_END_DATE,
    CLUSTERING_FEATURES_FILE,
    CLUSTERING_MANIFEST_FILE,
    CLUSTERING_MIN_ACTIVE_DAYS,
    CLUSTERING_MLFLOW_EXPERIMENT,
    CLUSTERING_MLFLOW_URI,
    CLUSTERING_SEED,
    CLUSTERING_START_DATE,
    FIGURES_DIR,
    PROJ_ROOT,
)
from mibici_ml.plots_clustering import graficar_mapa, graficar_perfiles

app = typer.Typer()

# Variables de uso del perfil (ver features_clustering.py). La capacidad no entra porque es
# un dato de infraestructura, no de cómo se usa la estación.
VARIABLES_USO = [
    "viajes_dia",
    "balance",
    "prop_manana",
    "prop_mediodia",
    "prop_tarde",
    "prop_noche",
    "prop_fin_semana",
    "prop_redondo",
    "duracion_mediana",
]
MODELOS = {
    "baseline": ["viajes_dia", "prop_fin_semana"],
    "candidato": VARIABLES_USO,
}
N_INIT = 10


def cargar_perfil():
    """Lee el perfil de estaciones y la versión de los datos del manifiesto."""
    if not CLUSTERING_FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"No existe {CLUSTERING_FEATURES_FILE}. "
            "Corre primero: python -m mibici_ml.features_clustering"
        )
    manifiesto = json.loads(CLUSTERING_MANIFEST_FILE.read_text(encoding="utf-8"))
    return pd.read_parquet(CLUSTERING_FEATURES_FILE), manifiesto["version_datos"]


def _git(*args):
    """Corre un comando de git en el repo y regresa su salida, o None si falla."""
    try:
        salida = subprocess.run(
            ["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True, check=True
        )
        return salida.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def crear_preprocesamiento():
    """Aplica logaritmo a viajes_dia y estandariza todas las variables.

    El logaritmo es porque viajes_dia va de 3 a 545 y sin él las estaciones con mucho uso
    dominarían las distancias. La estandarización deja todas las variables en la misma escala.
    """
    log_viajes = ColumnTransformer(
        [("log_viajes", FunctionTransformer(np.log1p), ["viajes_dia"])],
        remainder="passthrough",
    )
    return Pipeline([("log_viajes", log_viajes), ("escala", StandardScaler())])


def crear_modelo(k):
    """Pipeline completo: preprocesamiento y K-Means con k grupos."""
    kmeans = KMeans(n_clusters=k, n_init=N_INIT, random_state=CLUSTERING_SEED)
    return Pipeline([("preprocesamiento", crear_preprocesamiento()), ("kmeans", kmeans)])


def evaluar(x_eval, grupos):
    """Métricas internas del clustering, calculadas en el espacio de evaluación.

    silhouette: de -1 a 1, más alto es mejor; qué tan cerca está cada estación de su grupo
        comparado con el grupo más cercano.
    davies_bouldin: más bajo es mejor; qué tan compactos y separados están los grupos.
    calinski_harabasz: más alto es mejor; variación entre grupos contra variación dentro.
    grupo_mas_chico_pct: porcentaje de estaciones en el grupo más chico, para detectar
        grupos de muy pocas estaciones.
    """
    return {
        "silhouette": silhouette_score(x_eval, grupos),
        "davies_bouldin": davies_bouldin_score(x_eval, grupos),
        "calinski_harabasz": calinski_harabasz_score(x_eval, grupos),
        "grupo_mas_chico_pct": np.bincount(grupos).min() / len(grupos) * 100,
    }


def resumir_grupos(perfil, grupos):
    """Promedio de cada variable por grupo, para interpretar qué tipo de estaciones son."""
    resumen = perfil.assign(grupo=grupos).groupby("grupo")
    tabla = resumen[[*VARIABLES_USO, "capacidad"]].mean().round(3)
    tabla.insert(0, "n_estaciones", resumen.size())
    return tabla.reset_index()


def registrar_figuras(nombre_run, asignacion, x_eval, grupos):
    """Registra el mapa y el perfil de los grupos en MLflow y los guarda en reports/figures."""
    figuras = {
        "mapa_grupos.png": graficar_mapa(asignacion),
        "perfil_grupos.png": graficar_perfiles(x_eval, grupos, VARIABLES_USO),
    }
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for nombre, figura in figuras.items():
        mlflow.log_figure(figura, nombre)
        figura.savefig(FIGURES_DIR / f"{nombre_run}_{nombre}")


def entrenar_y_registrar(perfil, x_eval, modelo, k, version_datos):
    """Entrena un modelo con k grupos y registra todo en un run de MLflow."""
    variables = MODELOS[modelo]
    pipeline = crear_modelo(k)
    grupos = pipeline.fit_predict(perfil[variables])
    metricas = evaluar(x_eval, grupos)
    nombre_run = f"p3-{modelo}-kmeans-k{k}"

    with mlflow.start_run(run_name=nombre_run):
        mlflow.set_tags(
            {
                "equipo": "50",
                "pregunta": "P3",
                "git_commit": _git("rev-parse", "HEAD") or "desconocido",
                # Solo cuenta archivos que ya están en Git, no las figuras nuevas
                "git_cambios_sin_commit": bool(_git("status", "--porcelain", "-uno")),
            }
        )
        mlflow.log_params(
            {
                "modelo": modelo,
                "algoritmo": "KMeans",
                "k": k,
                "n_init": N_INIT,
                "semilla": CLUSTERING_SEED,
                "variables": ",".join(variables),
                "fecha_inicio": CLUSTERING_START_DATE,
                "fecha_fin": CLUSTERING_END_DATE,
                "version_datos": version_datos,
                "min_dias_activos": CLUSTERING_MIN_ACTIVE_DAYS,
                "n_estaciones": len(perfil),
            }
        )
        mlflow.log_metrics(metricas)

        columnas = ["estacion_id", "nombre", "zona", "latitud", "longitud"]
        asignacion = perfil[columnas].assign(grupo=grupos)
        mlflow.log_text(asignacion.to_csv(index=False), "grupos_estaciones.csv")
        mlflow.log_text(resumir_grupos(perfil, grupos).to_csv(index=False), "perfil_grupos.csv")
        mlflow.log_artifact(str(CLUSTERING_MANIFEST_FILE))
        registrar_figuras(nombre_run, asignacion, x_eval, grupos)
        mlflow.sklearn.log_model(pipeline, name="modelo", input_example=perfil[variables].head(5))

    texto = ", ".join(f"{nombre} {valor:.3f}" for nombre, valor in metricas.items())
    logger.info(f"{nombre_run}: {texto}")


def entrenar(modelos=("baseline", "candidato"), ks=(4,)):
    """Entrena cada modelo con cada k y registra un run por combinación.

    Es la función que se llama desde otros scripts (por ejemplo, un main que corra las tres
    preguntas). La línea de comandos de abajo solo le pasa los argumentos.
    """
    desconocidos = set(modelos) - set(MODELOS)
    if desconocidos:
        raise ValueError(f"Modelos no válidos: {desconocidos}. Usa {list(MODELOS)}")

    perfil, version_datos = cargar_perfil()
    x_eval = crear_preprocesamiento().fit_transform(perfil[VARIABLES_USO])

    mlflow.set_tracking_uri(CLUSTERING_MLFLOW_URI)
    mlflow.set_experiment(CLUSTERING_MLFLOW_EXPERIMENT)
    for modelo in modelos:
        for k in ks:
            entrenar_y_registrar(perfil, x_eval, modelo, k, version_datos)
    logger.success(f"Runs registrados en el experimento {CLUSTERING_MLFLOW_EXPERIMENT}")


@app.command()
def main(
    modelo: list[str] = typer.Option(["baseline", "candidato"], help="baseline o candidato"),
    k: list[int] = typer.Option([4], help="Número de grupos; se puede repetir"),
):
    """Línea de comandos: entrena los modelos y valores de k indicados."""
    entrenar(modelo, k)


if __name__ == "__main__":
    app()
