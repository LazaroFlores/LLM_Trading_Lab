# Experimento: test_experiment

Este experimento usa los mismos CSV schemas que el dataset histórico.

Archivos:
- `Experiments/test_experiment/csv_files/Daily Updates.csv`
- `Experiments/test_experiment/csv_files/Trade Log.csv`

Ejecutar mantenimiento/registro (paper/live, según tu uso):

```powershell
.\.venv311\Scripts\python.exe Experiments\chatgpt_micro-cap\scripts\processing\trading_script.py --data-dir "Experiments/test_experiment/csv_files" --starting-equity 10000.0
```

Recomendaciones (si usas este Daily Updates como universo/holdings):

```powershell
.\.venv311\Scripts\python.exe .\recommend.py --universe-from-daily --holdings-from-daily --daily-updates "Experiments/test_experiment/csv_files/Daily Updates.csv"
```
