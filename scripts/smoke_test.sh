#!/bin/sh
# Проверяет уже запущенный сервис: /health и один корректный /predict.
set -eu

BASE="${1:-http://127.0.0.1:8000}"
IMAGE="${2:-tests/fixtures/late_blight.jpg}"

i=0
while [ "$i" -lt 30 ]; do
  if curl -fsS "$BASE/health" > /tmp/plant-health.json 2>/dev/null; then
    break
  fi
  i=$((i + 1))
  sleep 1
done

health=$(curl -fsS "$BASE/health")
printf '%s\n' "$health"
printf '%s\n' "$health" | grep -q '"model_loaded":true'

curl -fsS "$BASE/docs" > /dev/null

predict=$(curl -fsS -F "file=@${IMAGE};type=image/jpeg" "$BASE/predict")
printf '%s\n' "$predict"
printf '%s\n' "$predict" | grep -q 'Tomato___Late_blight'
printf '%s\n' "smoke ok"
