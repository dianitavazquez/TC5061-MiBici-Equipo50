#!/usr/bin/env bash
set -euo pipefail

OUTPUT_FILE=${1:?Uso: monitor_resources.sh <archivo.csv> [intervalo_segundos]}
INTERVAL=${2:-2}

mkdir -p "$(dirname "$OUTPUT_FILE")"
echo "timestamp,container,cpu_percent,memory_usage,memory_percent" > "$OUTPUT_FILE"

while true; do
  timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  container_ids=$(
    {
      docker compose ps -q api mlflow
      docker ps -q --filter "name=^/mibici-locust-run$"
    } | sort -u | tr '\n' ' '
  )

  if [[ -n "${container_ids// }" ]]; then
    docker stats --no-stream \
      --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}' \
      $container_ids |
      while IFS= read -r metrics; do
        echo "$timestamp,$metrics" >> "$OUTPUT_FILE"
      done
  fi

  sleep "$INTERVAL"
done
