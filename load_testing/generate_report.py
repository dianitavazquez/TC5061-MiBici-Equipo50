"""Genera gráficas y resumen Markdown a partir de CSV de Locust y Docker."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def number(row, key, default=0.0):
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def aggregated_history(rows):
    aggregated = [row for row in rows if row.get("Name") == "Aggregated"]
    return aggregated or rows


def plot_performance(history, output_path):
    rows = aggregated_history(history)
    timestamps = [number(row, "Timestamp") for row in rows]
    start = min(timestamps) if timestamps else 0
    elapsed = [timestamp - start for timestamp in timestamps]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for percentile, label in (("50%", "p50"), ("95%", "p95"), ("99%", "p99")):
        axes[0].plot(elapsed, [number(row, percentile) for row in rows], label=label)
    axes[0].set_ylabel("Latencia (ms)")
    axes[0].set_title("Latencia del endpoint de predicción")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        elapsed,
        [number(row, "Requests/s") for row in rows],
        color="tab:green",
    )
    axes[1].set_ylabel("Peticiones/s")
    axes[1].set_title("Throughput")
    axes[1].grid(alpha=0.3)

    error_rates = []
    for row in rows:
        requests = number(row, "Total Request Count")
        failures = number(row, "Total Failure Count")
        error_rates.append((failures / requests * 100) if requests else 0)
    axes[2].plot(elapsed, error_rates, color="tab:red")
    axes[2].set_ylabel("Errores (%)")
    axes[2].set_xlabel("Tiempo transcurrido (s)")
    axes[2].set_title("Tasa de error acumulada")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def clean_percent(value):
    try:
        return float(value.replace("%", "").strip())
    except (AttributeError, ValueError):
        return 0.0


def plot_resources(rows, output_path):
    containers = sorted({row["container"] for row in rows})
    parsed_timestamps = [
        datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")) for row in rows
    ]
    start = min(parsed_timestamps)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for container in containers:
        selected = [row for row in rows if row["container"] == container]
        x_values = [
            (
                datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")) - start
            ).total_seconds()
            for row in selected
        ]
        axes[0].plot(
            x_values,
            [clean_percent(row["cpu_percent"]) for row in selected],
            label=container,
        )
        axes[1].plot(
            x_values,
            [clean_percent(row["memory_percent"]) for row in selected],
            label=container,
        )

    axes[0].set_ylabel("CPU (%)")
    axes[0].set_title("Uso de CPU por contenedor")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_ylabel("Memoria (%)")
    axes[1].set_xlabel("Tiempo aproximado (s)")
    axes[1].set_title("Uso de memoria por contenedor")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def find_aggregate(rows):
    for row in rows:
        if row.get("Name") == "Aggregated":
            return row
    raise ValueError("El CSV de estadísticas no contiene la fila Aggregated")


def resource_summary(rows):
    summary = {}
    for container in sorted({row["container"] for row in rows}):
        selected = [row for row in rows if row["container"] == container]
        cpu = [clean_percent(row["cpu_percent"]) for row in selected]
        memory = [clean_percent(row["memory_percent"]) for row in selected]
        summary[container] = {
            "cpu_avg": mean(cpu) if cpu else 0,
            "cpu_max": max(cpu, default=0),
            "memory_avg": mean(memory) if memory else 0,
            "memory_max": max(memory, default=0),
        }
    return summary


def write_report(args, aggregate, resources):
    requests = number(aggregate, "Request Count")
    failures = number(aggregate, "Failure Count")
    error_rate = failures / requests * 100 if requests else 0
    p95 = number(aggregate, "95%")
    verdict = (
        "PASA" if p95 <= args.p95_threshold and error_rate <= args.error_threshold else "NO PASA"
    )

    resource_rows = "\n".join(
        f"| `{container}` | {values['cpu_avg']:.2f}% | {values['cpu_max']:.2f}% | "
        f"{values['memory_avg']:.2f}% | {values['memory_max']:.2f}% |"
        for container, values in resources.items()
    )

    report = f"""# Reporte de prueba de carga MiBici

## Escenario

- Endpoint: `POST /predict`
- Usuarios concurrentes: **{args.users}**
- Tasa de arranque: **{args.spawn_rate} usuarios/s**
- Duración configurada: **{args.run_time}**
- Espera por usuario: **0.2 a 1.0 segundos**
- Datos: cinco estaciones tomadas del ejemplo de serving del modelo P2

## Resultados

| Métrica | Resultado |
|---|---:|
| Peticiones | {requests:.0f} |
| Fallas | {failures:.0f} |
| Tasa de error | {error_rate:.3f}% |
| Peticiones/s | {number(aggregate, "Requests/s"):.2f} |
| Latencia promedio | {number(aggregate, "Average Response Time"):.2f} ms |
| Latencia p50 | {number(aggregate, "50%"):.2f} ms |
| Latencia p95 | {p95:.2f} ms |
| Latencia p99 | {number(aggregate, "99%"):.2f} ms |

Objetivos provisionales: p95 ≤ {args.p95_threshold:.0f} ms y tasa de error ≤
{args.error_threshold:.2f}%. **Veredicto: {verdict}.**

![Latencia, throughput y errores](performance.png)

## Recursos de contenedores

| Contenedor | CPU promedio | CPU máxima | Memoria promedio | Memoria máxima |
|---|---:|---:|---:|---:|
{resource_rows}

![CPU y memoria](resources.png)

## Interpretación

Los resultados describen únicamente esta carga, esta máquina y la ruta local de red.
Antes de atribuir un límite a la API se debe comprobar que Docker Desktop y el contenedor
generador de Locust no estén saturados. El HTML de Locust y los CSV de esta carpeta
conservan el detalle para auditoría.
"""
    (args.output_dir / "README.md").write_text(report, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--spawn-rate", type=float, required=True)
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--p95-threshold", type=float, default=500)
    parser.add_argument("--error-threshold", type=float, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = read_csv(args.stats)
    history = read_csv(args.history)
    resources = read_csv(args.resources)
    if not history:
        raise ValueError("El CSV de historial de Locust está vacío")
    if not resources:
        raise ValueError("No se capturaron métricas de docker stats")
    aggregate = find_aggregate(stats)

    plot_performance(history, args.output_dir / "performance.png")
    plot_resources(resources, args.output_dir / "resources.png")
    write_report(args, aggregate, resource_summary(resources))


if __name__ == "__main__":
    main()
