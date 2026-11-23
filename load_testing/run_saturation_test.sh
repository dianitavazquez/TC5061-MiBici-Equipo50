#!/usr/bin/env bash
# Prueba de saturación de la API con 1 vCPU y 512 MiB: Locust sin espera en escalones de
# 1 a 64 usuarios y docker stats cada segundo. Los CSV que deja son los que lee mibici_ml/finops.py.
# Uso: bash load_testing/run_saturation_test.sh [carpeta_de_salida]
set -uo pipefail

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
SALIDA="${1:-$RAIZ/reports/load-tests/$(date -u +%Y%m%dT)-saturacion-1cpu}"
IMAGEN="${IMAGEN:-mibici-equipo50-api:latest}"
export MSYS_NO_PATHCONV=1

mkdir -p "$SALIDA"
docker rm -f finops-api finops-locust >/dev/null 2>&1
docker network create finops-net >/dev/null 2>&1
docker run -d --name finops-api --network finops-net --cpus=1 --memory=512m "$IMAGEN" >/dev/null

for _ in $(seq 1 60); do
  estado=$(docker inspect -f '{{.State.Health.Status}}' finops-api)
  [ "$estado" = "healthy" ] && break
  sleep 2
done
echo "API: $estado"

echo "timestamp,container,cpu_percent,memory_usage,memory_percent" > "$SALIDA/recursos.csv"
(
  while true; do
    instante=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}' \
      finops-api finops-locust 2>/dev/null \
      | while IFS= read -r linea; do echo "$instante,$linea" >> "$SALIDA/recursos.csv"; done
    sleep 1
  done
) &
monitor=$!

docker run --rm --name finops-locust --network finops-net \
  -v "$RAIZ/load_testing:/mnt/locust:ro" -v "$SALIDA:/out" \
  locustio/locust:2.32.10 -f /mnt/locust/locustfile_saturation.py --host http://finops-api:8000 \
  --headless --csv /out/saturacion --csv-full-history --only-summary

kill "$monitor"
wait "$monitor" 2>/dev/null
docker rm -f finops-api >/dev/null
docker network rm finops-net >/dev/null
echo "Resultados en $SALIDA"
