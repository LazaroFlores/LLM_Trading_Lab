# LLM Trading Lab - Descripción del proyecto

**Objetivo actual (2026+):** usar el análisis, datos y tooling de este repositorio para **diseñar y evaluar estrategias con intención de generar rendimientos**, de forma **reproducible y auditable**.

Esto implica:

- Definir señales/estrategias (principalmente en equities, con frecuencia diaria/semanal)
- Evaluarlas históricamente con métricas claras (retornos, drawdowns, comparación vs benchmarks, etc.)
- Mantener un registro consistente de datos/artefactos para poder auditar decisiones y resultados

## Origen (dataset y caso de estudio)

El repo nació como un experimento "forward-only" de micro-caps en EE. UU. donde un LLM (ChatGPT) gestionaba un portafolio bajo reglas estrictas. Ese material se conserva como **dataset/caso de estudio** y como base para extender el laboratorio.

## Qué hay dentro (hoy)

La carpeta más importante es `Experiments/chatgpt_micro-cap/`:

- `csv_files/`
  - `Daily Updates.csv`: estado diario del portafolio (incluye fila `TOTAL` con equity/caja).
  - `Trade Log.csv`: bitácora de operaciones (compras/ventas, PnL, motivo, etc.).
- `scripts/processing/trading_script.py`: script central para mantenimiento/registro del portafolio.
  - Descarga precios principalmente desde **Yahoo Finance (yfinance)** y usa **fallback a Stooq** (si está disponible).
  - Soporta modo reproducible "as-of date" (`--asof` o `ASOF_DATE=YYYY-MM-DD`).
- `graphing/`: scripts para generar gráficos (por default a `runs/YYYY-MM-DD/plots/`, que está ignorado por git).
- `evaluation/`: reporte y paper del estudio histórico (`evaluation_report.md`, `paper.pdf`).
- `collected_artifacts/`: reportes semanales ("Deep Research") y enlaces a conversaciones.

## Cómo se usa (rápido)

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar el script (usando el dataset del experimento):

```bash
python Experiments/chatgpt_micro-cap/scripts/processing/trading_script.py --data-dir Experiments/chatgpt_micro-cap/csv_files --starting-equity 100
```

Reproducibilidad por fecha:

```bash
python Experiments/chatgpt_micro-cap/scripts/processing/trading_script.py --data-dir Experiments/chatgpt_micro-cap/csv_files --asof 2025-08-15 --starting-equity 100 --skip
```

## Próximos pasos

Ver `TODO.md` para la hoja de ruta hacia un pipeline de estrategias (backtests, señales, comparables, y ejecución/operación controlada).

## Recomendaciones (estado actual)

Existe un recomendador cuantitativo simple en `recommend.py` que puede generar:

- Un Top N de oportunidades desde `config/universe.txt`
- Un seguimiento de holdings desde `config/holdings.txt` con señales BUY/SELL/HOLD basadas en estrategia long-only de canales + niveles de Fibonacci (pisos/techos)
- Un resumen de conteo de señales BUY/SELL/HOLD en cada corrida
- Un archivo de órdenes sugeridas (CSV) si el usuario confirma compras en modo interactivo
- Un backtest 1Y por holding para comparar estrategia vs buy&hold con capital inicial configurable (default: `100 USD`)
- Un trade log diario por ticker (`holdings_trade_log.csv`) con señal, posición, acción y equity
- Gráficos por holding con precio/canal/Fibonacci, marcas BUY/SELL/HOLD, conteo de señales y curva de equity con capital inicial/final y rendimiento en USD

Para experimentos propios, el repo incluye:

- `init_experiment.py`: inicializa un experimento nuevo con `Daily Updates.csv` y `Trade Log.csv`
- `paper_trade.py`: aplica `orders.csv` o convierte `holdings_trade_log.csv` a órdenes y actualiza los CSV del experimento (paper trading; long/short)
- `run_daily.ps1`: atajo diario que encadena `recommend.py` + `paper_trade.py` en una sola ejecución
- `run_daily.bat`: atajo para doble clic que llama `run_daily.ps1` con parámetros por defecto
