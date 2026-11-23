"""FinOps of the P2 API: measured capacity, timings and cost per inference in the cloud."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import time

from loguru import logger
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import mlflow.pyfunc
import numpy as np
import pandas as pd
import typer

from mibici_ml.config import FIGURES_DIR, MODELS_DIR, REFERENCES_DIR, REPORTS_DIR
from mibici_ml.plots import BACKGROUND, BORDER, GROUP_COLOR, SECONDARY_TEXT, TEXT

app = typer.Typer(no_args_is_help=True)

# Prueba de saturación: API con 1 vCPU y 512 MiB, Locust sin espera y escalones de 1 a 64
# usuarios de 40 s cada uno (load_testing/locustfile_saturation.py).
FINOPS_LOAD_TEST_DIR = REPORTS_DIR / "load-tests" / "20261122T-saturacion-1cpu"
FINOPS_API_CONTAINER = "finops-api"
FINOPS_MODEL_URI = MODELS_DIR / "serving" / "p2"
FINOPS_SUMMARY_DIR = REFERENCES_DIR / "finops"
FINOPS_FIGURE_DIR = FIGURES_DIR / "finops"
FINOPS_OUTPUT_NAME = "p2_finops"

P95_LIMIT_MS = 500  # mismo objetivo que la prueba de carga de 50 usuarios
TARGET_UTILIZATION = 0.7  # cada réplica trabaja al 70 % de su capacidad para aguantar picos
WARMUP_SECONDS = 10  # Locust calcula los percentiles "actuales" con una ventana de 10 s
HOURS_PER_MONTH = 730
STATIONS = 469  # estaciones del periodo actual (octubre de 2026)
REAL_DEMAND_RPS = STATIONS / 3600  # una predicción por estación cada hora
SERVICE_MEMORY_GIB = 0.5  # límite de la prueba; la API usó como máximo 236 MiB
# Prueba de 50 usuarios del equipo (reports/load-tests/20261122T-locust-50u/README.md).
REFERENCE_LOAD = {"rps": 78.44, "api_cpu_vcpu": 0.4318, "mlflow_memory_gib": 2.114}

# Precios públicos consultados el 2026-11-22. EC2: Linux bajo demanda en us-east-1 (archivo de
# precios de AWS publicado el 2026-11-21) y Spot del mismo día. Cloud Run: us-central1 con
# facturación por petición.
PRICE_DATE = "2026-11-22"
PRICE_SOURCES = {
    "ec2_on_demand": "https://aws.amazon.com/ec2/pricing/on-demand/",
    "ec2_spot": "https://aws.amazon.com/ec2/spot/pricing/",
    "t3_baseline": (
        "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/burstable-credits-baseline-concepts.html"
    ),
    "cloud_run": "https://cloud.google.com/run/pricing",
}


@dataclass(frozen=True)
class Instance:
    """EC2 instance price and size; burstable t3 instances include a baseline per vCPU."""

    name: str
    vcpus: int
    memory_gib: float
    usd_hour: float
    baseline: float | None = None
    spot_usd_hour: float | None = None


INSTANCES = {
    instance.name: instance
    for instance in (
        Instance("t3.micro", 2, 1, 0.0104, baseline=0.10),
        Instance("t3.small", 2, 2, 0.0208, baseline=0.20, spot_usd_hour=0.008),
        Instance("t3.medium", 2, 4, 0.0416, baseline=0.20, spot_usd_hour=0.0179),
        Instance("c7i.large", 2, 4, 0.08925),
    )
}
T3_SURPLUS_USD_VCPU_HOUR = 0.05  # modo unlimited: CPU arriba de la línea base
CLOUD_RUN = {
    "vcpu_second": 0.000024,
    "gib_second": 0.0000025,
    "million_requests": 0.40,
    "rounding_seconds": 0.1,
    "free_vcpu_seconds": 180_000,
    "free_gib_seconds": 360_000,
    "free_requests": 2_000_000,
}

SERIES_COLORS = (GROUP_COLOR, "#eb6834", "#1baf7a")
REFERENCE_COLOR = "#898781"
_MIB_PER_UNIT = {
    "B": 1 / 2**20,
    "KiB": 1 / 2**10,
    "kB": 1e3 / 2**20,
    "MiB": 1.0,
    "MB": 1e6 / 2**20,
    "GiB": 2.0**10,
    "GB": 1e9 / 2**20,
}


@app.callback()
def main():
    """Estimate the cost per inference of the P2 API from its load tests."""


def parse_mib(text: str) -> float:
    """Convert a docker stats size such as 210.6MiB to MiB."""
    match = re.fullmatch(r"\s*([\d.]+)\s*([A-Za-z]+)\s*", str(text))
    if not match or match.group(2) not in _MIB_PER_UNIT:
        raise ValueError(f"Tamaño de memoria no reconocido: {text!r}")
    return float(match.group(1)) * _MIB_PER_UNIT[match.group(2)]


def read_docker_stats(path: Path) -> pd.DataFrame:
    """Read the docker stats CSV; 100 % of CPU in docker stats is one vCPU."""
    frame = pd.read_csv(path)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp"], utc=True),
            "container": frame["container"],
            "cpu_vcpu": frame["cpu_percent"].str.rstrip("%").astype(float) / 100,
            "memory_mib": frame["memory_usage"].str.split("/").str[0].map(parse_mib),
        }
    )


def read_locust_history(path: Path) -> pd.DataFrame:
    """Read the aggregated per-second rows of a Locust stats history CSV."""
    frame = pd.read_csv(path, na_values="N/A")
    frame = frame[(frame["Name"] == "Aggregated") & (frame["User Count"] > 0)]
    frame = frame.dropna(subset=["50%", "95%"])
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["Timestamp"], unit="s", utc=True),
            "users": frame["User Count"].astype(int),
            "rps": frame["Requests/s"].astype(float),
            "failures_per_s": frame["Failures/s"].astype(float),
            "p50_ms": frame["50%"].astype(float),
            "p95_ms": frame["95%"].astype(float),
        }
    ).reset_index(drop=True)


def saturation_table(
    history: pd.DataFrame,
    stats: pd.DataFrame,
    container: str = FINOPS_API_CONTAINER,
    warmup_seconds: int = WARMUP_SECONDS,
    min_rows: int = 5,
) -> pd.DataFrame:
    """Summarize each user step with its throughput, latency and API resources."""
    api = stats[stats["container"] == container]
    if api.empty:
        raise ValueError(f"No hay filas del contenedor {container} en docker stats.")
    rows = []
    for users, step in history.groupby("users"):
        start = step["timestamp"].min() + pd.Timedelta(seconds=warmup_seconds)
        end = step["timestamp"].max()
        stable = step[step["timestamp"] >= start]
        resources = api[(api["timestamp"] >= start) & (api["timestamp"] <= end)]
        if len(stable) < min_rows or resources.empty:
            continue  # transiciones de un segundo entre escalones
        rps = float(stable["rps"].median())
        cpu = float(resources["cpu_vcpu"].mean())
        rows.append(
            {
                "users": int(users),
                "rps": rps,
                "p50_ms": float(stable["p50_ms"].median()),
                "p95_ms": float(stable["p95_ms"].median()),
                "failures_per_s": float(stable["failures_per_s"].max()),
                "cpu_vcpu": cpu,
                "memory_mib": float(resources["memory_mib"].max()),
                "cpu_ms_per_request": 1000 * cpu / rps,
            }
        )
    if not rows:
        raise ValueError("Ningún escalón tiene suficientes segundos estables.")
    return pd.DataFrame(rows)


def sustainable_capacity(table: pd.DataFrame, p95_limit_ms: float = P95_LIMIT_MS) -> dict:
    """Throughput of the busiest step that still meets the p95 objective without failures."""
    passing = table[(table["p95_ms"] <= p95_limit_ms) & (table["failures_per_s"] == 0)]
    if passing.empty:
        raise ValueError(f"Ningún escalón cumple p95 <= {p95_limit_ms} ms.")
    row = passing.loc[passing["users"].idxmax()]
    return {
        "users": int(row["users"]),
        "rps": float(row["rps"]),
        "p95_ms": float(row["p95_ms"]),
        "cpu_vcpu": float(row["cpu_vcpu"]),
        "cpu_ms_per_request": float(row["cpu_ms_per_request"]),
        "peak_rps": float(table["rps"].max()),
        "max_memory_mib": float(table["memory_mib"].max()),
    }


def ec2_hourly_cost(instance: Instance, cpu_vcpu: float, spot: bool = False) -> float:
    """Hourly price plus the unlimited-mode charge when a t3 runs above its baseline."""
    price = instance.spot_usd_hour if spot else instance.usd_hour
    if price is None:
        raise ValueError(f"No hay precio Spot para {instance.name}.")
    if instance.baseline is None:
        return price
    surplus = max(0.0, cpu_vcpu - instance.baseline * instance.vcpus)
    return price + surplus * T3_SURPLUS_USD_VCPU_HOUR


def replicas_needed(
    demand_rps: float, capacity_rps: float, target_utilization: float = TARGET_UTILIZATION
) -> int:
    """API replicas (one uvicorn worker each) needed to serve the demand with headroom."""
    return max(1, math.ceil(demand_rps / (capacity_rps * target_utilization)))


def _monthly_summary(usd_month: float, predictions_month: float, **details) -> dict:
    return {
        "usd_month": usd_month,
        "predictions_month": predictions_month,
        "usd_per_million": usd_month / predictions_month * 1e6,
        **details,
    }


def ec2_cost(
    instance: Instance,
    demand_rps: float,
    capacity: dict,
    memory_gib: float = SERVICE_MEMORY_GIB,
    spot: bool = False,
    cpu_vcpu: float | None = None,
) -> dict:
    """Monthly cost of always-on EC2 replicas serving the demand."""
    if memory_gib > instance.memory_gib:
        raise ValueError(f"{memory_gib:.2f} GiB no caben en {instance.name}.")
    replicas = replicas_needed(demand_rps, capacity["rps"])
    if cpu_vcpu is None:
        cpu_vcpu = demand_rps * capacity["cpu_ms_per_request"] / 1000 / replicas
    hourly = replicas * ec2_hourly_cost(instance, cpu_vcpu, spot)
    return _monthly_summary(
        hourly * HOURS_PER_MONTH,
        demand_rps * 3600 * HOURS_PER_MONTH,
        replicas=replicas,
        cpu_vcpu=cpu_vcpu,
    )


def _cloud_run_bill(
    instance_seconds: float,
    requests: float,
    predictions: float,
    memory_gib: float,
    free_tier: bool,
) -> dict:
    vcpu_seconds = instance_seconds
    gib_seconds = instance_seconds * memory_gib
    if free_tier:
        vcpu_seconds = max(0.0, vcpu_seconds - CLOUD_RUN["free_vcpu_seconds"])
        gib_seconds = max(0.0, gib_seconds - CLOUD_RUN["free_gib_seconds"])
        requests = max(0.0, requests - CLOUD_RUN["free_requests"])
    usd_month = (
        vcpu_seconds * CLOUD_RUN["vcpu_second"]
        + gib_seconds * CLOUD_RUN["gib_second"]
        + requests / 1e6 * CLOUD_RUN["million_requests"]
    )
    return _monthly_summary(usd_month, predictions, instance_seconds=instance_seconds)


def cloud_run_cost(
    demand_rps: float,
    capacity: dict,
    memory_gib: float = SERVICE_MEMORY_GIB,
    free_tier: bool = False,
) -> dict:
    """Monthly cost of the API on Cloud Run with request-based billing and scale to zero.

    Cloud Run bills an instance while it handles at least one request, rounded up to 100 ms.
    With Poisson arrivals the share of time covered by a billed request is 1 - exp(-rate * 0.1);
    when the demand needs more than one instance, the instances run at the target utilization.
    """
    busy = 1 - math.exp(-demand_rps * CLOUD_RUN["rounding_seconds"])
    instances = max(busy, demand_rps / (capacity["rps"] * TARGET_UTILIZATION))
    requests = demand_rps * 3600 * HOURS_PER_MONTH
    seconds = instances * 3600 * HOURS_PER_MONTH
    return _cloud_run_bill(seconds, requests, requests, memory_gib, free_tier)


def cloud_run_batch_cost(
    stations: int,
    cold_start_seconds: float,
    batch_seconds: float,
    memory_gib: float = SERVICE_MEMORY_GIB,
    free_tier: bool = False,
) -> dict:
    """Monthly cost of predicting every station in one hourly call that starts from zero."""
    rounding = CLOUD_RUN["rounding_seconds"]
    billed = math.ceil(round((cold_start_seconds + batch_seconds) / rounding, 6)) * rounding
    runs = HOURS_PER_MONTH
    return _cloud_run_bill(billed * runs, runs, stations * runs, memory_gib, free_tier)


def station_batch(example_path: Path, stations: int = STATIONS) -> pd.DataFrame:
    """Repeat the model input example until there is one row per station."""
    example = json.loads(Path(example_path).read_text(encoding="utf-8"))
    frame = pd.DataFrame(example["data"], columns=example["columns"])
    batch = frame.iloc[np.arange(stations) % len(frame)].reset_index(drop=True)
    return batch.astype({"capacity": "int64", "station_id": "int64", "is_weekend": "int32"})


def time_predictions(model, frame: pd.DataFrame, repeats: int = 50) -> dict:
    """Median time to predict the whole batch and a single row with the same model."""

    def median_seconds(data: pd.DataFrame) -> float:
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model.predict(data)
            times.append(time.perf_counter() - start)
        return float(np.median(times))

    model.predict(frame)  # la primera llamada calienta cachés
    batch_seconds = median_seconds(frame)
    single_seconds = median_seconds(frame.iloc[:1])
    return {
        "rows": len(frame),
        "batch_seconds": batch_seconds,
        "single_row_seconds": single_seconds,
        "speedup_per_prediction": single_seconds * len(frame) / batch_seconds,
    }


def cold_start_seconds(path: Path) -> float:
    """Median seconds from creating the API container until /health answers."""
    runs = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(np.median([run["hasta_health_s"] for run in runs]))


def build_scenarios(capacity: dict, cold_start: float, timing: dict) -> pd.DataFrame:
    """Cost per inference for the current deployment, the proposed ones and higher loads."""
    real = REAL_DEMAND_RPS
    stack_memory = SERVICE_MEMORY_GIB + REFERENCE_LOAD["mlflow_memory_gib"]
    busy = capacity["rps"] * TARGET_UTILIZATION
    t3_small, t3_medium = INSTANCES["t3.small"], INSTANCES["t3.medium"]
    batch_seconds = timing["batch_seconds"] * STATIONS / timing["rows"]
    scenarios = [
        (
            "Compose actual: API + MLflow en t3.medium",
            real,
            ec2_cost(t3_medium, real, capacity, memory_gib=stack_memory),
        ),
        ("Solo la API en t3.micro", real, ec2_cost(INSTANCES["t3.micro"], real, capacity)),
        ("Solo la API en t3.small Spot", real, ec2_cost(t3_small, real, capacity, spot=True)),
        ("Cloud Run por petición", real, cloud_run_cost(real, capacity)),
        (
            "Cloud Run por lotes cada hora",
            real,
            cloud_run_batch_cost(STATIONS, cold_start, batch_seconds),
        ),
        (
            "Prueba de 50 usuarios en t3.small",
            REFERENCE_LOAD["rps"],
            ec2_cost(
                t3_small, REFERENCE_LOAD["rps"], capacity, cpu_vcpu=REFERENCE_LOAD["api_cpu_vcpu"]
            ),
        ),
        (
            "Prueba de 50 usuarios en Cloud Run",
            REFERENCE_LOAD["rps"],
            cloud_run_cost(REFERENCE_LOAD["rps"], capacity),
        ),
        ("Réplica al 70 % en t3.small", busy, ec2_cost(t3_small, busy, capacity)),
        ("Réplica al 70 % en c7i.large", busy, ec2_cost(INSTANCES["c7i.large"], busy, capacity)),
    ]
    rows = []
    for name, demand, cost in scenarios:
        free = None
        if name.startswith("Cloud Run"):
            free = (
                cloud_run_batch_cost(STATIONS, cold_start, batch_seconds, free_tier=True)
                if "lotes" in name
                else cloud_run_cost(demand, capacity, free_tier=True)
            )["usd_month"]
        rows.append(
            {
                "scenario": name,
                "demand_rps": demand,
                "replicas": cost.get("replicas"),
                "usd_month": cost["usd_month"],
                "usd_month_free_tier": free,
                "predictions_month": cost["predictions_month"],
                "usd_per_million": cost["usd_per_million"],
            }
        )
    return pd.DataFrame(rows)


def cost_curve(capacity: dict, demands: np.ndarray | None = None) -> pd.DataFrame:
    """Cost per million predictions as the demand grows, for three deployments."""
    demands = np.logspace(-2, 3, 181) if demands is None else demands
    stack_memory = SERVICE_MEMORY_GIB + REFERENCE_LOAD["mlflow_memory_gib"]
    rows = []
    for demand in demands:
        rows.append(
            {
                "demand_rps": float(demand),
                "API + MLflow en t3.medium": ec2_cost(
                    INSTANCES["t3.medium"], demand, capacity, memory_gib=stack_memory
                )["usd_per_million"],
                "Solo la API en t3.micro": ec2_cost(INSTANCES["t3.micro"], demand, capacity)[
                    "usd_per_million"
                ],
                "Cloud Run por petición": cloud_run_cost(demand, capacity)["usd_per_million"],
            }
        )
    return pd.DataFrame(rows)


def _style_axis(axis, grid: str = "y") -> None:
    axis.set_facecolor(BACKGROUND)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(BORDER)
    axis.tick_params(colors=SECONDARY_TEXT, labelsize=9)
    axis.grid(axis=grid, which="major", color=BORDER, linewidth=0.8)
    axis.set_axisbelow(True)


def plot_saturation(table: pd.DataFrame, capacity: dict, path: Path) -> Path:
    """Save throughput, latency and CPU of the API for each step of users."""
    figure = Figure(figsize=(11, 3.6), facecolor=BACKGROUND, layout="constrained")
    axes = figure.subplots(1, 3)
    users = table["users"]
    panels = [
        (axes[0], "Peticiones por segundo", [("rps", "Peticiones/s", SERIES_COLORS[0])]),
        (
            axes[1],
            "Latencia (ms)",
            [("p50_ms", "p50", SERIES_COLORS[0]), ("p95_ms", "p95", SERIES_COLORS[1])],
        ),
        (axes[2], "CPU de la API (% de 1 vCPU)", [("cpu_percent", "CPU", SERIES_COLORS[0])]),
    ]
    table = table.assign(cpu_percent=table["cpu_vcpu"] * 100)
    for axis, title, series in panels:
        _style_axis(axis)
        for column, label, color in series:
            axis.plot(users, table[column], color=color, linewidth=2, marker="o", markersize=5)
            if len(series) > 1:
                axis.annotate(
                    label,
                    (users.iloc[-1], table[column].iloc[-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                    color=SECONDARY_TEXT,
                )
        axis.set_xscale("log", base=2)
        axis.set_xticks(users, [str(value) for value in users])
        axis.set_xlabel("Usuarios simultáneos", color=SECONDARY_TEXT, fontsize=9)
        axis.set_title(title, loc="left", color=TEXT, fontsize=10)
        axis.set_ylim(bottom=0)
    axes[1].axhline(P95_LIMIT_MS, color=REFERENCE_COLOR, linewidth=1, linestyle="--")
    axes[1].text(
        users.iloc[0],
        P95_LIMIT_MS,
        f" objetivo p95 de {P95_LIMIT_MS} ms",
        va="bottom",
        fontsize=8,
        color=SECONDARY_TEXT,
    )
    axes[1].set_xlim(right=users.iloc[-1] * 1.6)
    axes[2].set_ylim(0, 110)
    axes[0].annotate(
        f"{capacity['rps']:.0f} req/s con {capacity['users']} usuarios",
        (capacity["users"], capacity["rps"]),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color=SECONDARY_TEXT,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, facecolor=BACKGROUND)
    return path


def plot_cost_curve(curve: pd.DataFrame, path: Path) -> Path:
    """Save the cost per million predictions against the demand on log scales."""
    figure = Figure(figsize=(9, 4.4), facecolor=BACKGROUND, layout="constrained")
    axis = figure.subplots()
    _style_axis(axis, grid="both")
    series = [column for column in curve.columns if column != "demand_rps"]
    for column, color in zip(series, SERIES_COLORS):
        axis.plot(curve["demand_rps"], curve[column], color=color, linewidth=2, label=column)
        axis.annotate(
            column,
            (curve["demand_rps"].iloc[-1], curve[column].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=SECONDARY_TEXT,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    for scale in (axis.xaxis, axis.yaxis):
        scale.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.10g}"))
    top = curve[series].to_numpy().max() * 3
    for demand, label in [
        (REAL_DEMAND_RPS, "demanda real de MiBici"),
        (REFERENCE_LOAD["rps"], "prueba de 50 usuarios"),
    ]:
        axis.axvline(demand, color=REFERENCE_COLOR, linewidth=1, linestyle="--")
        axis.text(demand, top, f" {label}", va="top", fontsize=8, color=SECONDARY_TEXT)
    axis.set_ylim(top=top)
    axis.set_xlim(curve["demand_rps"].min(), curve["demand_rps"].max() * 12)
    axis.set_xlabel("Demanda (peticiones por segundo)", color=SECONDARY_TEXT, fontsize=9)
    axis.set_ylabel("USD por millón de predicciones", color=SECONDARY_TEXT, fontsize=9)
    axis.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=TEXT)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, facecolor=BACKGROUND)
    return path


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def run_finops(
    load_test_dir: Path = FINOPS_LOAD_TEST_DIR,
    container: str = FINOPS_API_CONTAINER,
    model_uri: Path = FINOPS_MODEL_URI,
    summary_dir: Path = FINOPS_SUMMARY_DIR,
    figure_dir: Path = FINOPS_FIGURE_DIR,
    output_name: str = FINOPS_OUTPUT_NAME,
    timing: dict | None = None,
) -> dict:
    """Build the saturation table, the cost scenarios, the figures and a JSON summary."""
    load_test_dir = Path(load_test_dir)
    history = read_locust_history(load_test_dir / "saturacion_stats_history.csv")
    stats = read_docker_stats(load_test_dir / "recursos.csv")
    table = saturation_table(history, stats, container)
    capacity = sustainable_capacity(table)
    cold_start = cold_start_seconds(load_test_dir / "arranque_frio.json")
    if timing is None:
        model = mlflow.pyfunc.load_model(str(model_uri))
        timing = time_predictions(model, station_batch(Path(model_uri) / "input_example.json"))
    scenarios = build_scenarios(capacity, cold_start, timing)

    summary_dir, figure_dir = Path(summary_dir), Path(figure_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(summary_dir / f"{output_name}_saturacion.csv", index=False, float_format="%.4f")
    scenarios.to_csv(
        summary_dir / f"{output_name}_escenarios.csv", index=False, float_format="%.6g"
    )
    plot_saturation(table, capacity, figure_dir / f"{output_name}_saturacion.png")
    plot_cost_curve(cost_curve(capacity), figure_dir / f"{output_name}_costos.png")

    summary = _json_ready(
        {
            "price_date": PRICE_DATE,
            "price_sources": PRICE_SOURCES,
            "prices": {
                "ec2_us_east_1": {name: asdict(item) for name, item in INSTANCES.items()},
                "t3_surplus_usd_vcpu_hour": T3_SURPLUS_USD_VCPU_HOUR,
                "cloud_run_us_central1": CLOUD_RUN,
            },
            "assumptions": {
                "p95_limit_ms": P95_LIMIT_MS,
                "target_utilization": TARGET_UTILIZATION,
                "hours_per_month": HOURS_PER_MONTH,
                "stations": STATIONS,
                "real_demand_rps": REAL_DEMAND_RPS,
                "service_memory_gib": SERVICE_MEMORY_GIB,
                "reference_load": REFERENCE_LOAD,
            },
            "capacity": capacity,
            "cold_start_seconds": cold_start,
            "prediction_timing": timing,
            "saturation": table.to_dict(orient="records"),
            "scenarios": scenarios.to_dict(orient="records"),
        }
    )
    (summary_dir / f"{output_name}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


@app.command()
def run(
    load_test_dir: Path = FINOPS_LOAD_TEST_DIR,
    container: str = FINOPS_API_CONTAINER,
    model_uri: Path = FINOPS_MODEL_URI,
    summary_dir: Path = FINOPS_SUMMARY_DIR,
    figure_dir: Path = FINOPS_FIGURE_DIR,
    output_name: str = FINOPS_OUTPUT_NAME,
):
    """Compute capacity, timings and cost per inference of the P2 API."""
    summary = run_finops(load_test_dir, container, model_uri, summary_dir, figure_dir, output_name)
    capacity = summary["capacity"]
    logger.info(
        f"Capacidad sostenible: {capacity['rps']:.1f} req/s con {capacity['users']} usuarios "
        f"(p95 {capacity['p95_ms']:.0f} ms, {capacity['cpu_ms_per_request']:.2f} ms de CPU)"
    )
    for row in summary["scenarios"]:
        logger.info(
            f"{row['scenario']}: {row['usd_month']:.2f} USD/mes, "
            f"{row['usd_per_million']:.3f} USD por millón"
        )


if __name__ == "__main__":
    app()
