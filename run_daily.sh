#!/bin/bash
set -e

# Default values
EXPERIMENT=${1:-test_experiment}
TOP=${2:-0}
HOLDINGS_PATH=${3:-config/holdings.txt}
HOLDINGS_LOOKBACK_YEARS=${4:-5}
BACKTEST_DAYS=${5:-252}
INITIAL_CAPITAL=${6:-100}
OPEN_NOTIONAL=${7:-250}
SLIPPAGE_BPS=${8:-10}
FEE=${9:-0}
RUN_TAG=${10:-""}

DATA_DIR="Experiments/$EXPERIMENT/csv_files"
DAILY_CSV="$DATA_DIR/Daily Updates.csv"
TRADE_CSV="$DATA_DIR/Trade Log.csv"

# Calculate AsOf (Today or Friday if weekend)
if [ -n "$ASOF_DATE" ]; then
  ASOF="$ASOF_DATE"
else
  DOW=$(date +%u)
  if [ "$DOW" -eq 6 ]; then
    ASOF=$(date -d "yesterday" +%F)
  elif [ "$DOW" -eq 7 ]; then
    ASOF=$(date -d "2 days ago" +%F)
  else
    ASOF=$(date +%F)
  fi
fi

RUN_DIR="runs/$ASOF"
if [ -n "$RUN_TAG" ]; then
  RUN_DIR="$RUN_DIR/$RUN_TAG"
fi
HOLDINGS_TRADE_LOG="$RUN_DIR/holdings_trade_log.csv"

echo "AsOf: $ASOF"
echo "Experimento: $EXPERIMENT"
echo "Data dir: $DATA_DIR"

if [ ! -f "$DAILY_CSV" ]; then
  echo "Error: No existe $DAILY_CSV. Inicializa con init_experiment.py --name $EXPERIMENT"
  exit 1
fi

# Build recommend command
CMD_RECOMMEND="python recommend.py --non-interactive --asof $ASOF --top $TOP --holdings $HOLDINGS_PATH --holdings-lookback-years $HOLDINGS_LOOKBACK_YEARS --backtest-days $BACKTEST_DAYS --initial-capital $INITIAL_CAPITAL"

if [ -n "$RUN_TAG" ]; then
  CMD_RECOMMEND="$CMD_RECOMMEND --run-tag $RUN_TAG"
fi

# Add plot holdings if requested (currently hardcoded off in script but could be added)
# CMD_RECOMMEND="$CMD_RECOMMEND --plot-holdings"

echo "[1/2] Ejecutando recommend.py ..."
# Execute in subshell to capture exit code properly with set -e
(source .venv/bin/activate && $CMD_RECOMMEND)

if [ ! -f "$HOLDINGS_TRADE_LOG" ]; then
  echo "Error: No se generó $HOLDINGS_TRADE_LOG"
  exit 1
fi

# Build paper_trade command
CMD_PAPER="python paper_trade.py --data-dir $DATA_DIR --holdings-trade-log $HOLDINGS_TRADE_LOG --holdings-log-mode latest --open-notional $OPEN_NOTIONAL --fill next_open --slippage-bps $SLIPPAGE_BPS --fee $FEE"

echo "[2/2] Ejecutando paper_trade.py ..."
(source .venv/bin/activate && $CMD_PAPER)

echo "Listo."
echo "- Holdings trade log: $HOLDINGS_TRADE_LOG"
echo "- Daily Updates: $DAILY_CSV"
echo "- Trade Log: $TRADE_CSV"
