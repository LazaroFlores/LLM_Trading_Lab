﻿# TODO - LLM Trading Lab (objetivo: rendimientos)

Este archivo lista el trabajo pendiente para evolucionar el repo hacia un pipeline de investigación/ejecución orientado a retornos, manteniendo reproducibilidad y auditabilidad.

## Prioridad alta (próximas iteraciones)

- Definir un "núcleo" de evaluación: interfaz de estrategia, calendario, universo, costos (slippage/fees) y métricas estándar.
- Estandarizar el "data layer": descarga/caché, control de ajustes (splits/dividendos), validación de gaps y manejo de tickers delist.
- Normalizar paths/entrypoints:
  - Alinear documentacion/Makefile/workflows con la estructura real bajo `Experiments/chatgpt_micro-cap/`.
- Añadir un modo batch/research (no interactivo) para producir reportes comparables entre estrategias.
- Evolucionar `recommend.py`:
  - [x] Soportar sizing por riesgo (implementado via `confidence_weight` técnico + news sentiment).
  - Añadir backtest walk-forward de la señal usada para el Top N (evitar sobreajuste).
  - Registrar decisiones/señales con versionado (parámetros y datos as-of).
  - Calibrar estrategia de canales + Fibonacci por activo/regimen (hoy los resultados 1Y son mixtos en holdings).
  - Evaluar reintroducción de módulo short separado (actualmente la señal de holdings está en modo long-only táctico).
  - Añadir reglas de confirmación para reducir sobreoperación en rangos laterales.
  - Añadir reporte consolidado con ranking visual de holdings (PNL, maxDD, trades, hit-rate) para comparar rápidamente.
- Evolucionar `paper_trade.py`:
  - Costos realistas: comisiones por broker, slippage por liquidez, borrow fees para shorts, hard-to-borrow.
  - Validaciones de riesgo: límites por posición, exposición neta/bruta, drawdown stop.
  - Evitar duplicados de forma robusta (run_id por ejecución, hashing de órdenes).
  - Soportar parciales / no fills y órdenes limit.

## Prioridad media

- Convertir scripts de `graphing/` en un módulo reutilizable con outputs parametrizables (por estrategia y por periodo).
- Incorporar benchmarks configurables y persistencia de configuraciones (por ejemplo `tickers.json` donde aplique).
- Generar reportes "de una corrida" (tablas + gráficos) con nombre/versionado por fecha.

## Control de performance (cuando exista backtest)

- Mantener tiempos razonables de corrida semanal; documentar en `README.md` (sección "Performance evolution").
- Preferir operaciones vectorizadas/batched y reuso de datos (evitar recomputar/recargar).

## Nota

- El experimento histórico (CSV + artefactos) se preserva como dataset. Cualquier extensión "hasta hoy" debería hacerse en una copia de datos o en un experimento nuevo para no reescribir el histórico.
