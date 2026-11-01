
from pathlib import Path

root = Path.cwd()
train_path = root / "mibici_ml" / "modeling" / "train.py"
if not train_path.exists():
    raise SystemExit("No encuentro mibici_ml/modeling/train.py. Ejecuta este script desde la raíz del repositorio.")

source = train_path.read_text(encoding="utf-8")
if "P1_MODELS = {" in source and "p1-model-search" in source:
    raise SystemExit("Parece que la implementación nueva de P1 ya está aplicada; no hice cambios.")

source = source.replace(
    "from sklearn.ensemble import HistGradientBoostingRegressor",
    "from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor",
)
source = source.replace(
    "import mlflow.sklearn\n",
    "import mlflow.sklearn\nfrom mlflow import MlflowClient\nfrom mlflow.exceptions import MlflowException\nfrom mlflow.models import infer_signature\n",
)

start = source.index('@app.command("train-p1")')
end = source.index("\n\nP3_USAGE_FEATURES =", start)

replacement = r'''
P1_REGISTERED_MODEL_NAME = "mibici-equipo50-p1-demanda"


def build_p1_models() -> dict[str, object]:
    return {
        "p1-baseline-median": DummyRegressor(strategy="median"),
        "p1-hist-gradient-boosting": HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.1, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, random_state=42,
        ),
        "p1-random-forest": RandomForestRegressor(
            n_estimators=40, max_depth=20, min_samples_leaf=2,
            n_jobs=-1, random_state=42,
        ),
        "p1-extra-trees": ExtraTreesRegressor(
            n_estimators=40, max_depth=20, min_samples_leaf=2,
            n_jobs=-1, random_state=42,
        ),
    }


def _deterministic_sample(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0:
        raise ValueError("Sample row limits must be positive.")
    if len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=seed).sort_index()


@app.command("train-p1")
def train_p1(
    features_path: Path = P1_FEATURES_FILE,
    manifest_path: Path = P1_MANIFEST_FILE,
    model_path: Path = P1_MODEL_FILE,
    test_fraction: float = 0.2,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    max_train_rows: int = 250_000,
    max_test_rows: int = 100_000,
):
    """Run nested P1 regression experiments and register the MAE winner."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if max_train_rows <= 0 or max_test_rows <= 0:
        raise ValueError("max_train_rows and max_test_rows must be positive.")
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    logger.info("Reading P1 features from {}", features_path)
    df = pd.read_parquet(features_path)
    required = set(FEATURE_COLUMNS + ["timestamp", "target_departures"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[FEATURE_COLUMNS + ["target_departures"]].isna().any().any():
        raise ValueError("Features or target contain missing values.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    timestamps = sorted(df["timestamp"].unique())
    split_index = int(len(timestamps) * (1 - test_fraction))
    if split_index <= 0 or split_index >= len(timestamps):
        raise ValueError("Not enough timestamps for the requested split.")
    cutoff = pd.Timestamp(timestamps[split_index])
    train_df = df[df["timestamp"] < cutoff].copy()
    test_df = df[df["timestamp"] >= cutoff].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("The chronological split produced an empty partition.")

    # Bounded, deterministic samples keep tree ensembles practical on this dataset.
    train_sample = _deterministic_sample(train_df, max_train_rows, 42)
    test_sample = _deterministic_sample(test_df, max_test_rows, 43)
    X_train, y_train = train_sample[FEATURE_COLUMNS], train_sample["target_departures"]
    X_test, y_test = test_sample[FEATURE_COLUMNS], test_sample["target_departures"]

    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    dataset_version = manifest.get("dataset_sha256", "unknown")
    commit = git_commit()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    models = build_p1_models()
    results = []
    base_params = {
        "dataset_version": dataset_version,
        "features_file": features_path.name,
        "manifest_file": manifest_path.name,
        "train_start": str(train_df["timestamp"].min()),
        "train_end": str(train_df["timestamp"].max()),
        "test_start": str(test_df["timestamp"].min()),
        "test_end": str(test_df["timestamp"].max()),
        "split_strategy": "chronological",
        "test_fraction": test_fraction,
        "split_cutoff": cutoff.isoformat(),
        "random_state": 42,
        "n_features": len(FEATURE_COLUMNS),
        "train_rows_available": len(train_df),
        "test_rows_available": len(test_df),
        "train_rows_used": len(train_sample),
        "test_rows_used": len(test_sample),
        "max_train_rows": max_train_rows,
        "max_test_rows": max_test_rows,
        "sampling_policy": "deterministic_random_sample_after_chronological_split",
    }

    logger.info(
        "P1 split: {:,} train / {:,} test rows; sampled {:,} / {:,}; cutoff {}",
        len(train_df), len(test_df), len(train_sample), len(test_sample), cutoff,
    )

    with mlflow.start_run(run_name="p1-model-search") as parent_run:
        mlflow.set_tags({
            "team": "50", "question": "P1", "task": "regression",
            "git_commit": commit, "dataset_version": dataset_version,
            "experiment_role": "parent_search",
        })
        mlflow.log_params({
            **base_params, "candidate_count": len(models),
            "selection_metric": "test_mae",
            "registered_model_name": P1_REGISTERED_MODEL_NAME,
        })
        mlflow.log_artifact(str(manifest_path), artifact_path="data_metadata")

        for model_name, model in models.items():
            logger.info("Training nested run: {}", model_name)
            with mlflow.start_run(run_name=model_name, nested=True) as child_run:
                mlflow.set_tags({
                    "team": "50", "question": "P1", "task": "regression",
                    "git_commit": commit, "dataset_version": dataset_version,
                    "experiment_role": "candidate",
                })
                params = dict(base_params)
                params["model_name"] = model_name
                params.update({
                    f"model__{key}": value
                    for key, value in model.get_params(deep=False).items()
                    if value is None or isinstance(value, (str, int, float, bool))
                })
                mlflow.log_params(params)

                model.fit(X_train, y_train)
                metrics, predictions = evaluate_model(model, X_test, y_test)
                mlflow.log_metrics({
                    "test_mae": metrics["mae"], "test_rmse": metrics["rmse"],
                })
                plot_path = FIGURES_DIR / f"{model_name}_p1.png"
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                save_regression_scatter(y_test, predictions, model_name, plot_path)
                mlflow.log_artifact(str(plot_path), artifact_path="plots")

                input_example = X_train.head(5)
                signature = infer_signature(input_example, model.predict(input_example).clip(min=0))
                model_info = mlflow.sklearn.log_model(
                    model, name="model", signature=signature, input_example=input_example,
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )
                results.append({
                    "model": model_name, "mae": metrics["mae"], "rmse": metrics["rmse"],
                    "run_id": child_run.info.run_id, "model_uri": model_info.model_uri,
                    "model_object": model,
                })
                logger.info("{}: MAE={:.4f}, RMSE={:.4f}", model_name, metrics["mae"], metrics["rmse"])

        comparison = pd.DataFrame([
            {"model": item["model"], "mae": item["mae"], "rmse": item["rmse"], "run_id": item["run_id"]}
            for item in results
        ]).sort_values(["mae", "rmse"], ascending=True)
        comparison_path = P1_COMPARISON_FILE
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(comparison_path, index=False)
        mlflow.log_artifact(str(comparison_path), artifact_path="evaluation")
        mlflow.log_metrics({
            "best_test_mae": float(comparison.iloc[0]["mae"]),
            "best_test_rmse": float(comparison.iloc[0]["rmse"]),
        })

        winner_name = str(comparison.iloc[0]["model"])
        winner = next(item for item in results if item["model"] == winner_name)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(winner["model_object"], model_path)

        client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        try:
            client.create_registered_model(
                P1_REGISTERED_MODEL_NAME,
                description="P1: regresión para pronosticar salidas por estación-hora en MiBici.",
                tags={"team": "50", "question": "P1", "selection_metric": "test_mae"},
            )
        except MlflowException:
            # The registered model may already exist.
            pass
        registered_version = mlflow.register_model(
            model_uri=winner["model_uri"], name=P1_REGISTERED_MODEL_NAME,
            tags={"selection_metric": "test_mae", "winner_run_id": winner["run_id"]},
        )
        client.set_registered_model_alias(
            P1_REGISTERED_MODEL_NAME, "champion", str(registered_version.version)
        )
        mlflow.set_tag("winner_model", winner_name)
        mlflow.set_tag("registered_model_version", str(registered_version.version))

    logger.success("P1 comparison saved to {}", P1_COMPARISON_FILE)
    logger.success(
        "Champion: {} (MAE={:.4f}, RMSE={:.4f}), registered version {}",
        winner_name, winner["mae"], winner["rmse"], registered_version.version,
    )
    return comparison
'''

source = source[:start] + replacement + source[end:]
train_path.write_text(source, encoding="utf-8")

mlproject = r"""
name: mibici-p1

entry_points:
  ingest:
    command: "python -m mibici_ml.dataset prepare"

  features:
    command: "python -m mibici_ml.features build-p1"

  train:
    parameters:
      test_fraction:
        type: float
        default: 0.2
      max_train_rows:
        type: int
        default: 250000
      max_test_rows:
        type: int
        default: 100000
    command: "python -m mibici_ml.modeling.train train-p1 --test-fraction {test_fraction} --max-train-rows {max_train_rows} --max-test-rows {max_test_rows}"
"""
(root / "MLproject").write_text(mlproject.strip() + "\n", encoding="utf-8")
print("Actualicé mibici_ml/modeling/train.py y creé MLproject.")
print("Revisa los cambios con: git diff --check && git diff --stat")
