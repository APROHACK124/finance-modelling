#!/usr/bin/env bash
set -euo pipefail

# === Ajustes ===
PROJECT_ROOT="/home/aprohack/Desktop/all_folders/Investings_project/app"
VENV_PY="$PROJECT_ROOT/investenv/bin/python"
LOG_DIR="$PROJECT_ROOT/logs"
LOCK_FILE="/tmp/investings_daily.lock"

# Crea directorio de logs si no existe
mkdir -p "$LOG_DIR"

# Fecha para logs
STAMP="$(date +'%Y-%m-%d_%H-%M-%S')"
LOG_FILE="$LOG_DIR/run_${STAMP}.log"

# Exporta PYTHONPATH para evitar líos de imports si ejecutas .py directos
export PYTHONPATH="$PROJECT_ROOT"

# Cambia al root del proyecto
cd "$PROJECT_ROOT"

# Usa flock para que no se solape si cron dispara de nuevo mientras corre
{
  echo "=== RUN START $(date -Iseconds) ==="

  # Opción A: ejecutar como módulos (recomendado)
  
  $VENV_PY -m database_tier1             2>&1 | sed 's/^/[database_tier1] /'
  $VENV_PY -m calculate_indicators       2>&1 | sed 's/^/[calculate_indicators] /'
  $VENV_PY -m tier2_data_scrapper        2>&1 | sed 's/^/[tier2_data_scrapper] /'
  $VENV_PY -m fetch_news_data            2>&1 | sed 's/^/[fetch_news_data] /'
  $VENV_PY -m fetch_economic_data        2>&1 | sed 's/^/[fetch_economic_data] /'
  $VENV_PY -m fetch_reddit_data          2>&1 | sed 's/^/[fetch_reddit_data] /'
  $VENV_PY -m preprocess_data            2>&1 | sed 's/^/[preprocess_data] /'
  $VENV_PY -m llm_sentiment_analysis_v2     2>&1 | sed 's/^/[llm_sentiment_analysis_v2] /'
  

  echo "=== RUN END $(date -Iseconds) ==="
} | tee -a "$LOG_FILE" | tee -a "$PROJECT_ROOT/cron.log" \
  | awk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}' \
  > /dev/null

# Con flock: bloquea toda la ejecución para evitar solapamientos
# Uso: flock -n LOCKFILE command
# Para usarlo, invoca este script con flock fuera (ver crontab más abajo)
