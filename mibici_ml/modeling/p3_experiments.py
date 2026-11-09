"""Experimentación de P3 en MLflow: búsqueda de k con nested runs y candidatos.

Todas las corridas usan el mismo perfil de estaciones y se evalúan en el mismo espacio que
la entrega anterior (las 9 variables de uso, con logaritmo en viajes_dia y estandarizadas),
así que sus métricas se pueden comparar contra el baseline. Además de las métricas internas,
cada corrida mide la estabilidad de los grupos: entrena con submuestras de estaciones y
compara sus grupos contra los del modelo completo con el índice de Rand ajustado (ARI).

Uso (desde la raíz del repo):
    python -m mibici_ml.modeling.p3_experiments buscar
    python -m mibici_ml.modeling.p3_experiments comparar
    python -m mibici_ml.modeling.p3_experiments entrenar --modelo candidato-1
    python -m mibici_ml.modeling.p3_experiments registrar
"""

import json
import time

from loguru import logger
from matplotlib.figure import Figure
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
import typer

from mibici_ml.config import (
    CLUSTERING_CANDIDATES,
    CLUSTERING_CHALLENGER,
    CLUSTERING_CHAMPION,
    CLUSTERING_END_DATE,
    CLUSTERING_FEATURES_FILE,
    CLUSTERING_MANIFEST_FILE,
    CLUSTERING_MIN_ACTIVE_DAYS,
    CLUSTERING_REGISTERED_MODEL,
    CLUSTERING_REGISTERED_MODEL_DESCRIPTION,
    CLUSTERING_SEARCH_KS,
    CLUSTERING_SEED,
    CLUSTERING_STABILITY_FRACTION,
    CLUSTERING_STABILITY_RUNS,
    CLUSTERING_START_DATE,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
)
from mibici_ml.modeling.train import (
    P3_MODELS,
    P3_USAGE_FEATURES,
    build_p3_preprocessor,
    evaluate_clustering,
    git_commit,
    summarize_clusters,
)
from mibici_ml.plots import plot_cluster_map, plot_cluster_profiles

app = typer.Typer(no_args_is_help=True)

# Conjuntos de variables. "uso_sin_mediodia" quita una franja porque las cuatro suman 1.
VARIABLES = {
    "basico": P3_MODELS["baseline"],
    "uso": P3_USAGE_FEATURES,
    "uso_sin_mediodia": [v for v in P3_USAGE_FEATURES if v != "prop_mediodia"],
}
FAMILIAS = {"kmeans": "kmeans", "gmm": "gaussian_mixture"}
COLUMNAS_ESTACION = ["estacion_id", "nombre", "zona", "latitud", "longitud"]


def preparar_mlflow():
    """Apunta MLflow al servidor o base de datos del proyecto y al experimento del equipo."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def cargar_datos():
    """Lee el perfil de estaciones, el manifiesto y arma el espacio de evaluación."""
    if not CLUSTERING_FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"No existe {CLUSTERING_FEATURES_FILE}. Corre: python -m mibici_ml.features build-p3"
        )
    perfil = pd.read_parquet(CLUSTERING_FEATURES_FILE)
    manifiesto = json.loads(CLUSTERING_MANIFEST_FILE.read_text(encoding="utf-8"))
    x_eval = build_p3_preprocessor().fit_transform(perfil[P3_USAGE_FEATURES])
    return perfil, manifiesto, x_eval


def crear_modelo(config):
    """Pipeline con el preprocesamiento de P3 y el algoritmo indicado en la configuración."""
    if config["algoritmo"] == "kmeans":
        estimador = KMeans(
            n_clusters=config["k"], n_init=config["n_init"], random_state=CLUSTERING_SEED
        )
    elif config["algoritmo"] == "gmm":
        estimador = GaussianMixture(
            n_components=config["k"],
            covariance_type=config.get("covariance_type", "full"),
            n_init=config["n_init"],
            random_state=CLUSTERING_SEED,
        )
    else:
        raise ValueError(f"Algoritmo no válido: {config['algoritmo']}. Usa {list(FAMILIAS)}")
    return Pipeline([("preprocesamiento", build_p3_preprocessor()), ("modelo", estimador)])


def medir_estabilidad(config, datos):
    """ARI promedio y mínimo entre los grupos del modelo completo y los de submuestras.

    Un valor cercano a 1 indica que los grupos casi no cambian al quitar estaciones, es decir,
    que no dependen de unas cuantas estaciones en particular.
    """
    referencia = crear_modelo(config).fit(datos).predict(datos)
    generador = np.random.default_rng(CLUSTERING_SEED)
    tamano = int(len(datos) * CLUSTERING_STABILITY_FRACTION)
    valores = []
    for _ in range(CLUSTERING_STABILITY_RUNS):
        indices = generador.choice(len(datos), size=tamano, replace=False)
        modelo = crear_modelo(config).fit(datos.iloc[indices])
        valores.append(adjusted_rand_score(referencia, modelo.predict(datos)))
    return float(np.mean(valores)), float(np.min(valores))


def registrar_artefactos(modelo, perfil, variables, grupos, x_eval, config, manifiesto):
    """Guarda modelo, datos, gráficas y configuración en subcarpetas del run."""
    entrada = perfil[variables]
    asignacion = perfil[COLUMNAS_ESTACION].assign(grupo=grupos)

    mlflow.log_figure(plot_cluster_map(asignacion), "plots/mapa_grupos.png")
    mlflow.log_figure(
        plot_cluster_profiles(x_eval, grupos, P3_USAGE_FEATURES), "plots/perfil_grupos.png"
    )
    mlflow.log_artifact(str(CLUSTERING_MANIFEST_FILE), artifact_path="data")
    mlflow.log_text(asignacion.to_csv(index=False), "data/grupos_estaciones.csv")
    mlflow.log_text(
        summarize_clusters(perfil, grupos).to_csv(index=False), "data/perfil_grupos.csv"
    )
    mlflow.log_dict(
        {
            **config,
            "lista_variables": variables,
            "fecha_inicio": CLUSTERING_START_DATE,
            "fecha_fin": CLUSTERING_END_DATE,
            "version_datos": manifiesto["version_datos"],
            "semilla": CLUSTERING_SEED,
        },
        "config/configuracion.json",
    )
    dataset = mlflow.data.from_pandas(
        perfil[["estacion_id", *variables]],
        source=str(CLUSTERING_FEATURES_FILE),
        name="perfil_estaciones_p3",
    )
    mlflow.log_input(dataset, context="training")
    mlflow.sklearn.log_model(
        modelo,
        name="modelo",
        signature=infer_signature(entrada, grupos),
        input_example=entrada.head(5),
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )


def entrenar_y_registrar(nombre, config, datos, etapa, anidado=False, artefactos=True):
    """Entrena una configuración, mide sus métricas y la registra como un run de MLflow.

    etapa es el valor de la etiqueta stage_of_work: baseline, search o candidate.
    """
    perfil, manifiesto, x_eval = datos
    variables = VARIABLES[config["variables"]]
    modelo = crear_modelo(config)
    inicio = time.perf_counter()
    grupos = modelo.fit(perfil[variables]).predict(perfil[variables])
    tiempo = time.perf_counter() - inicio
    estabilidad_media, estabilidad_min = medir_estabilidad(config, perfil[variables])
    metricas = {
        **evaluate_clustering(x_eval, grupos),
        "estabilidad_ari_media": estabilidad_media,
        "estabilidad_ari_min": estabilidad_min,
        "tiempo_entrenamiento_s": tiempo,
    }
    nombre_run = f"p3-{nombre}-{config['algoritmo']}-k{config['k']}"

    with mlflow.start_run(run_name=nombre_run, nested=anidado) as run:
        mlflow.set_tags(
            {
                "team": "50",
                "question": "P3",
                "task": "clustering",
                "model_family": FAMILIAS[config["algoritmo"]],
                "stage_of_work": etapa,
                "variant": nombre,
                "git_commit": git_commit(),
            }
        )
        mlflow.log_params(
            {
                **config,
                "semilla": CLUSTERING_SEED,
                "fecha_inicio": CLUSTERING_START_DATE,
                "fecha_fin": CLUSTERING_END_DATE,
                "version_datos": manifiesto["version_datos"],
                "min_dias_activos": CLUSTERING_MIN_ACTIVE_DAYS,
                "n_estaciones": len(perfil),
                "estabilidad_repeticiones": CLUSTERING_STABILITY_RUNS,
                "estabilidad_fraccion": CLUSTERING_STABILITY_FRACTION,
            }
        )
        mlflow.log_metrics(metricas)
        if artefactos:
            registrar_artefactos(modelo, perfil, variables, grupos, x_eval, config, manifiesto)

    resumen = ", ".join(f"{k} {v:.3f}" for k, v in metricas.items())
    logger.info(f"{nombre_run}: {resumen}")
    return run.info.run_id, metricas


def graficar_busqueda(tabla):
    """Silhouette y estabilidad por valor de k, en dos paneles con el mismo eje x."""
    figura = Figure(figsize=(9, 3.6), dpi=150, layout="constrained")
    ejes = figura.subplots(1, 2, sharex=True)
    paneles = [("silhouette", "Silhouette"), ("estabilidad_ari_media", "Estabilidad (ARI)")]
    for eje, (columna, titulo) in zip(ejes, paneles):
        eje.plot(tabla["k"], tabla[columna], marker="o", color="#2a78d6", linewidth=2)
        eje.set_title(titulo, loc="left")
        eje.set_xlabel("Número de grupos (k)")
        eje.grid(alpha=0.3)
        eje.spines[["top", "right"]].set_visible(False)
    return figura


def buscar_k():
    """Búsqueda de k para K-Means con las 9 variables: un run padre y un hijo por cada k."""
    preparar_mlflow()
    datos = cargar_datos()
    with mlflow.start_run(run_name="p3-busqueda-kmeans-k"):
        mlflow.set_tags(
            {
                "team": "50",
                "question": "P3",
                "task": "clustering",
                "model_family": "kmeans",
                "stage_of_work": "search",
                "git_commit": git_commit(),
            }
        )
        mlflow.log_params({"ks": ",".join(map(str, CLUSTERING_SEARCH_KS)), "variables": "uso"})
        filas = []
        for k in CLUSTERING_SEARCH_KS:
            config = {"algoritmo": "kmeans", "variables": "uso", "k": k, "n_init": 10}
            _, metricas = entrenar_y_registrar(
                "busqueda", config, datos, etapa="search", anidado=True, artefactos=False
            )
            filas.append({"k": k, **metricas})
        tabla = pd.DataFrame(filas)
        mejor = tabla.loc[tabla["silhouette"].idxmax()]
        mlflow.log_metrics({"mejor_k": mejor["k"], "mejor_silhouette": mejor["silhouette"]})
        mlflow.log_text(tabla.to_csv(index=False), "data/resultados_busqueda.csv")
        mlflow.log_figure(graficar_busqueda(tabla), "plots/metricas_por_k.png")
    logger.success(f"Búsqueda terminada; mejor k por silhouette: {int(mejor['k'])}")


def entrenar_modelo(nombre, hiperparametros=None):
    """Entrena una de las configuraciones de config.py, con hiperparámetros opcionales."""
    if nombre not in CLUSTERING_CANDIDATES:
        raise ValueError(f"Modelo no válido: {nombre}. Usa {list(CLUSTERING_CANDIDATES)}")
    config = {**CLUSTERING_CANDIDATES[nombre], **(hiperparametros or {})}
    etapa = "baseline" if nombre == "baseline" else "candidate"
    preparar_mlflow()
    return entrenar_y_registrar(nombre, config, cargar_datos(), etapa=etapa)


def buscar_ultimo_run(cliente, nombre):
    """Run más reciente de una configuración de P3, buscado por su etiqueta variant."""
    experimento = cliente.get_experiment_by_name(MLFLOW_EXPERIMENT)
    runs = cliente.search_runs(
        [experimento.experiment_id],
        filter_string=f"tags.question = 'P3' and tags.variant = '{nombre}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No hay runs de {nombre}. Corre primero el comando comparar.")
    return runs[0]


def registrar_version(cliente, run, alias):
    """Registra el modelo de un run, guarda su linaje como etiquetas y le asigna un alias."""
    modelos = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"source_run_id = '{run.info.run_id}'",
        output_format="list",
    )
    version = mlflow.register_model(modelos[0].model_uri, CLUSTERING_REGISTERED_MODEL)
    linaje = {
        "experimento": MLFLOW_EXPERIMENT,
        "run_origen": run.info.run_name,
        "run_id": run.info.run_id,
        "git_commit": run.data.tags["git_commit"],
        "version_datos": run.data.params["version_datos"],
        "silhouette": f"{run.data.metrics['silhouette']:.3f}",
        "estabilidad_ari_media": f"{run.data.metrics['estabilidad_ari_media']:.3f}",
    }
    for clave, valor in linaje.items():
        cliente.set_model_version_tag(CLUSTERING_REGISTERED_MODEL, version.version, clave, valor)
    cliente.update_model_version(
        CLUSTERING_REGISTERED_MODEL,
        version.version,
        description=(
            f"{alias}: run {run.info.run_name}, commit {linaje['git_commit'][:7]}, "
            f"datos {linaje['version_datos']}"
        ),
    )
    cliente.set_registered_model_alias(CLUSTERING_REGISTERED_MODEL, alias, version.version)
    return version


def registrar_modelos(campeon=CLUSTERING_CHAMPION, retador=CLUSTERING_CHALLENGER):
    """Registra el champion y el challenger, y comprueba que el champion carga por su alias."""
    preparar_mlflow()
    cliente = MlflowClient()
    for nombre, alias in [(campeon, "champion"), (retador, "challenger")]:
        version = registrar_version(cliente, buscar_ultimo_run(cliente, nombre), alias)
        logger.info(f"{nombre}: versión {version.version} con alias {alias}")
    cliente.update_registered_model(
        CLUSTERING_REGISTERED_MODEL, description=CLUSTERING_REGISTERED_MODEL_DESCRIPTION
    )
    cliente.set_registered_model_tag(CLUSTERING_REGISTERED_MODEL, "team", "50")
    cliente.set_registered_model_tag(CLUSTERING_REGISTERED_MODEL, "question", "P3")

    uri = f"models:/{CLUSTERING_REGISTERED_MODEL}@champion"
    modelo = mlflow.sklearn.load_model(uri)
    perfil, _, _ = cargar_datos()
    grupos = modelo.predict(perfil[modelo.feature_names_in_].head(5))
    logger.success(f"{uri} cargado; grupos de las primeras 5 estaciones: {grupos.tolist()}")


@app.command("buscar")
def buscar_command():
    """Búsqueda de k con nested runs."""
    buscar_k()


@app.command("entrenar")
def entrenar_command(
    modelo: str = typer.Option(..., help="baseline, candidato-1, candidato-2 o candidato-3"),
    hiperparametros: str = typer.Option("{}", help='JSON, por ejemplo {"k": 5}'),
):
    """Entrena y registra un solo modelo."""
    entrenar_modelo(modelo, json.loads(hiperparametros))


@app.command("comparar")
def comparar_command():
    """Entrena y registra el baseline y los tres candidatos."""
    for nombre in CLUSTERING_CANDIDATES:
        entrenar_modelo(nombre)


@app.command("registrar")
def registrar_command(
    campeon: str = typer.Option(CLUSTERING_CHAMPION, help="Modelo para el alias champion"),
    retador: str = typer.Option(CLUSTERING_CHALLENGER, help="Modelo para el alias challenger"),
):
    """Registra los modelos elegidos en el Model Registry y verifica la carga."""
    registrar_modelos(campeon, retador)


if __name__ == "__main__":
    app()
