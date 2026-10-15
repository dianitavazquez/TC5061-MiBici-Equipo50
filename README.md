# MiBici-Equipo50

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Propuesta y base reproducible de proyecto MiBici

## Clasificador P2

El baseline predice presión de demanda por falta de bicicletas y de anclajes para cada
estación en la hora siguiente. El diseño, las fuentes, la estrategia temporal y las
métricas están documentados en [docs/p2-classifier-design.md](docs/p2-classifier-design.md).

```bash
# Requiere Python 3.12. No utilizar un entorno creado con Python 3.9.
python3.12 -m venv .venv
source .venv/bin/activate
make requirements
python -m mibici_ml.dataset download-drive
python -m mibici_ml.dataset profile
python -m mibici_ml.modeling.train p2-baseline
make mlflow-ui
```

Si `mlflow.db` reporta una revisión de Alembic desconocida, la terminal está usando
otra versión de MLflow. Verifica `python --version` y `python -m mlflow --version`;
deben indicar Python 3.12 y MLflow 3.17.1, respectivamente. No borres `mlflow.db`, pues
contiene los experimentos registrados.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici_ml and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mibici_ml   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici_ml a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
