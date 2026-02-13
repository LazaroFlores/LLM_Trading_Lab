# LLM Trading Lab

Objetivo actual: convertir este repositorio en un **laboratorio de investigación cuantitativa** orientado a **generar rendimientos** mediante:

- Definición de señales/estrategias (principalmente equity)
- Evaluación histórica reproducible (métricas, drawdowns, benchmarks)
- Iteración auditable (sin reescribir el pasado; agregando análisis y estrategias)

## Origen (dataset y caso de estudio)

El repositorio comenzó como un experimento "forward-only" de micro-caps donde un LLM (ChatGPT) gestionaba un portafolio bajo reglas estrictas. Ese material se conserva como **dataset/caso de estudio** y como base de tooling.

- Paper del experimento: `Experiments/chatgpt_micro-cap/evaluation/paper.pdf`

## Aviso

Este repositorio es **investigación** y no constituye asesoramiento financiero. Cualquier objetivo de "generar rendimientos" implica riesgo real de pérdida.

## Estructura (actual)

```text
LLM_Trading_Lab/
|-- Description.md
|-- TODO.md
|-- requirements.txt
|-- runs/                         # outputs locales (ignorados por git)
|
|-- Experiments/
|   |-- chatgpt_micro-cap/
|       |-- csv_files/
|       |   |-- Daily Updates.csv
|       |   |-- Trade Log.csv
|       |
|       |-- scripts/
|       |   |-- processing/
|       |   |   |-- trading_script.py
|       |   |   |-- ProcessPortfolio.py
|       |   |-- metrics/
|       |
|       |-- graphing/
|       |-- images/               # imágenes "fijas" del reporte histórico
|       |-- evaluation/
|       |-- collected_artifacts/
```

## Inicio rápido (Windows)

Recomendado: Python 3.11 (los paquetes fijados en `requirements.txt` no apuntan a Python 3.9).

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

## Crear un experimento propio (CSV)

Para llevar tus propios `Daily Updates.csv` y `Trade Log.csv` (separado del dataset histórico), inicializa un experimento nuevo:

```powershell
.\.venv311\Scripts\python.exe .\init_experiment.py --name mi_experimento --starting-equity 10000
```

## Flujo diario recomendado (paper trading)

Objetivo: generar recomendaciones, confirmar órdenes y dejar todo registrado en tus CSV (sin ejecutar trades reales).

1) Genera recomendaciones y (opcional) órdenes:

```powershell
.\.venv311\Scripts\python.exe .\recommend.py --top 5 --universe-from-daily --daily-updates "Experiments\mi_experimento\csv_files\Daily Updates.csv"
```

- Si corres en modo interactivo, te preguntará si quieres generar órdenes `BUY` (Top y/o holdings con `BUY`) y también `SELL` (venta total) para holdings con señal `SELL`.
- Esto genera `runs/YYYY-MM-DD/orders.csv` y `runs/YYYY-MM-DD/snapshot.json`.

2) Aplica las órdenes y actualiza `Daily Updates.csv` + `Trade Log.csv` del experimento:

```powershell
.\.venv311\Scripts\python.exe .\paper_trade.py --data-dir Experiments\mi_experimento\csv_files --orders "runs\YYYY-MM-DD\orders.csv" --fill next_open --slippage-bps 10 --fee 0
```

2b) Alternativa automática desde `holdings_trade_log.csv` (sin `orders.csv` manual):

```powershell
.\.venv311\Scripts\python.exe .\paper_trade.py --data-dir Experiments\mi_experimento\csv_files --holdings-trade-log "runs\YYYY-MM-DD\holdings_trade_log.csv" --holdings-log-mode latest --open-notional 250 --fill next_open --slippage-bps 10 --fee 0
```

Notas:
- `paper_trade.py` soporta long y short. Si envías una orden `SELL` sin posición previa, se interpreta como entrada short (paper).
- Por ahora `recommend.py` no propone shorts automáticamente; si los quieres, se pueden añadir reglas/estrategia.
- `--fill next_open` (default) simula ejecución en la siguiente sesión a la fecha `asof` del order. Alternativa: `--fill next_close`.
- Con `--holdings-trade-log`, `paper_trade.py` convierte acciones del log (`BUY`, `SELL_TO_CLOSE`, `SHORT`, `COVER`, `FLIP_*`) en órdenes ejecutables.
- `--holdings-log-mode latest` usa el último día del log, `date` usa `--holdings-log-date`, y `all` recorre todas las fechas del log.

3) Atajo todo-en-uno (recomendación + ejecución paper):

```powershell
.\run_daily.ps1 -Experiment mi_experimento -Top 0 -OpenNotional 250 -HoldingsLogMode latest -Fill next_open -SlippageBps 10 -Fee 0
```

Notas del script:
- Usa `recommend.py` en modo no interactivo y luego `paper_trade.py` con `holdings_trade_log.csv`.
- Si hoy cae en fin de semana, usa viernes como `asof`.
- Puedes activar gráficos con `-PlotHoldings`.

3b) Atajo para doble clic (CMD/Windows):

```bat
.\run_daily.bat
```

Notas del `.bat`:
- Ejecuta `run_daily.ps1` con parámetros por defecto.
- Si le pasas argumentos, los reenvía tal cual a `run_daily.ps1`.

Ejecutar el script de mantenimiento/registro usando el dataset del experimento:

```powershell
.\.venv311\Scripts\python.exe Experiments\chatgpt_micro-cap\scripts\processing\trading_script.py --data-dir Experiments\chatgpt_micro-cap\csv_files --starting-equity 100
```

Modo reproducible por fecha ("trata esta fecha como hoy"):

```powershell
.\.venv311\Scripts\python.exe Experiments\chatgpt_micro-cap\scripts\processing\trading_script.py --data-dir Experiments\chatgpt_micro-cap\csv_files --asof 2025-08-15 --starting-equity 100 --skip
```

Generar un gráfico (ejemplo):

```powershell
.\.venv311\Scripts\python.exe Experiments\chatgpt_micro-cap\graphing\equity_vs_baseline.py
```

## Recomendaciones (Top 5 + holdings)

Este repo incluye un recomendador cuantitativo simple que:

- Genera un **Top N** de tickers candidatos desde un universo (`config/universe.txt`)
- Genera acciones **BUY/SELL/HOLD** para tus holdings (`config/holdings.txt`) usando estrategia **long-only táctica** de **pisos/techos/canales + Fibonacci** (`SELL` = salida defensiva)
- (Opcional) Te pregunta si quieres **generar órdenes** de compra; guarda un CSV (no ejecuta trades)
- Muestra un backtest rapido de 1 ano por holding (estrategia vs buy&hold) con capital inicial configurable (default: `100 USD`)
- Muestra resumen de conteo de señales en consola (`BUY/SELL/HOLD`) para holdings

Configuración:
- Edita `config/universe.txt` y `config/holdings.txt`

Ejecutar:

```powershell
.\.venv311\Scripts\python.exe .\recommend.py --top 5
```

Solo holdings (sin Top del universo), usando 3 anos de lookback y capital inicial de 100 USD:

```powershell
.\.venv311\Scripts\python.exe .\recommend.py --top 0 --holdings .\config\holdings.txt --holdings-lookback-years 3 --backtest-days 252 --initial-capital 100
```

Generar grafico por holding (precio+canal+Fibonacci con marcas BUY/SELL/HOLD, conteo de señales y equity curve en USD):

```powershell
.\.venv311\Scripts\python.exe .\recommend.py --top 0 --holdings .\config\holdings.txt --holdings-lookback-years 3 --backtest-days 252 --initial-capital 100 --plot-holdings
```

Salida de graficos por default:
- `runs/YYYY-MM-DD/plots/*.png`

Usar un universo externo (por ejemplo, el de CodexTrader) con prefijos tipo `NYSE:` / `NASDAQ:` / `BMV:`:

```powershell
.\.venv311\Scripts\python.exe .\recommend.py --top 5 --universe "C:\Users\fer_f\PycharmProjects\CodexTrader\data\tickers.txt"
```

Usar el histórico del experimento como universo/holdings (derivado del último día en `Daily Updates.csv`):

```powershell
.\.venv311\Scripts\python.exe .\recommend.py --top 5 --universe-from-daily --holdings-from-daily
```

Nota: en el dataset histórico incluido, el último día (`2025-12-26`) solo tiene la fila `TOTAL`, así que `--holdings-from-daily` devolverá vacío a menos que uses tu propio `Daily Updates.csv`.

Outputs:
- `runs/YYYY-MM-DD/snapshot.json`
- `runs/YYYY-MM-DD/orders.csv` (si confirmas compras en modo interactivo)
- `runs/YYYY-MM-DD/holdings_trade_log.csv` (registro diario por ticker de señal, posición, acción y equity)

Campos nuevos en `snapshot.json`:
- `holdings_signal_counts`: conteo de señales `BUY/SELL/HOLD`
- `holdings_portfolio_backtest_1y`: resumen agregado del portafolio (capital inicial/final y retornos estrategia vs buy&hold)

## Qué se registra

El histórico base para análisis vive en:

- `Experiments/chatgpt_micro-cap/csv_files/Daily Updates.csv`: estado diario (incluye fila `TOTAL` con equity/caja).
- `Experiments/chatgpt_micro-cap/csv_files/Trade Log.csv`: bitácora de trades (compras/ventas, PnL, motivo, etc.).

## Artefactos de investigación (experimento histórico)

- Índice de research semanal: `Experiments/chatgpt_micro-cap/collected_artifacts/deep_research_index.md`
- Chats (links): `Experiments/chatgpt_micro-cap/collected_artifacts/chats.md`

## Performance evolution

Pendiente de estandarizar. La meta es mantener tiempos de ejecución "semanales" dentro de límites prácticos (ver `TODO.md`).

## Contribuir

Guía local: `Other/CONTRIBUTING.md`
