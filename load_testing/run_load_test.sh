#!/usr/bin/env bash
set -euo pipefail

USERS=${USERS:-50}
SPAWN_RATE=${SPAWN_RATE:-5}
RUN_TIME=${RUN_TIME:-2m}
STATS_INTERVAL=${STATS_INTERVAL:-2}
P95_THRESHOLD_MS=${P95_THRESHOLD_MS:-500}
ERROR_RATE_THRESHOLD=${ERROR_RATE_THRESHOLD:-1}
RUN_ID=${RUN_ID:-$(date -u +"%Y%m%dT%H%M%SZ")}
KEEP_SERVICES=${KEEP_SERVICES:-0}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

HOST_RESULTS_DIR="reports/load-tests/$RUN_ID"
CONTAINER_RESULTS_DIR="/reports/$RUN_ID"
mkdir -p "$HOST_RESULTS_DIR"
chmod 0777 "$HOST_RESULTS_DIR"

monitor_pid=""
cleanup() {
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  if [[ "$KEEP_SERVICES" != "1" ]]; then
    docker compose --profile load-test down
  fi
}
trap cleanup EXIT INT TERM

echo "Construyendo y levantando la API..."
docker compose --profile load-test build api locust
docker compose up -d --wait api

echo "Monitoreando recursos en $HOST_RESULTS_DIR/resources.csv"
./load_testing/monitor_resources.sh \
  "$HOST_RESULTS_DIR/resources.csv" "$STATS_INTERVAL" &
monitor_pid=$!

echo "Ejecutando Locust: users=$USERS spawn_rate=$SPAWN_RATE run_time=$RUN_TIME"
set +e
docker compose --profile load-test run --rm --no-deps \
  --name mibici-locust-run locust \
  -f /mnt/locust/locustfile.py \
  --host http://api:8000 \
  --headless \
  --users "$USERS" \
  --spawn-rate "$SPAWN_RATE" \
  --run-time "$RUN_TIME" \
  --csv "$CONTAINER_RESULTS_DIR/locust" \
  --html "$CONTAINER_RESULTS_DIR/locust-report.html" \
  --only-summary
locust_status=$?
set -e

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=""

echo "Generando gráficas y reporte Markdown..."
docker compose --profile load-test run --rm --no-deps --entrypoint python locust \
  /mnt/locust/generate_report.py \
  --stats "$CONTAINER_RESULTS_DIR/locust_stats.csv" \
  --history "$CONTAINER_RESULTS_DIR/locust_stats_history.csv" \
  --resources "$CONTAINER_RESULTS_DIR/resources.csv" \
  --output-dir "$CONTAINER_RESULTS_DIR" \
  --users "$USERS" \
  --spawn-rate "$SPAWN_RATE" \
  --run-time "$RUN_TIME" \
  --p95-threshold "$P95_THRESHOLD_MS" \
  --error-threshold "$ERROR_RATE_THRESHOLD"

echo "Reporte generado en $HOST_RESULTS_DIR/README.md"
exit "$locust_status"
