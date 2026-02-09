from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


DAILY_UPDATES_HEADER = [
    "Date",
    "Ticker",
    "Shares",
    "Buy Price",
    "Cost Basis",
    "Stop Loss",
    "Current Price",
    "Total Value",
    "PnL",
    "Action",
    "Cash Balance",
    "Total Equity",
]

TRADE_LOG_HEADER = [
    "Date",
    "Ticker",
    "Shares Bought",
    "Buy Price",
    "Cost Basis",
    "PnL",
    "Reason",
    "Shares Sold",
    "Sell Price",
]


def _coerce_date(d: str | None) -> pd.Timestamp:
    if d is None:
        out = pd.Timestamp(datetime.now().date())
    else:
        out = pd.Timestamp(d)
    out = out.normalize()
    # Si cae en fin de semana, usar viernes (consistente con otros scripts)
    if out.dayofweek == 5:
        out = out - pd.Timedelta(days=1)
    elif out.dayofweek == 6:
        out = out - pd.Timedelta(days=2)
    return out


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inicializa un experimento nuevo (CSV schema compatible).")
    parser.add_argument("--name", required=True, help="Nombre del experimento (crea Experiments/<name>/csv_files).")
    parser.add_argument("--starting-equity", required=True, type=float, help="Capital inicial (USD/MXN según tu experimento).")
    parser.add_argument("--start-date", default=None, help="Fecha YYYY-MM-DD (default: hoy; fin de semana -> viernes).")
    args = parser.parse_args()

    if args.starting_equity <= 0:
        raise ValueError("--starting-equity debe ser positivo.")

    exp_dir = Path("Experiments") / args.name
    csv_dir = exp_dir / "csv_files"
    daily_path = csv_dir / "Daily Updates.csv"
    trade_path = csv_dir / "Trade Log.csv"

    if exp_dir.exists():
        raise FileExistsError(f"Ya existe: {exp_dir}")

    start = _coerce_date(args.start_date)

    # Seed: solo TOTAL (sin posiciones) + cash = equity
    total_row = {
        "Date": start.date().isoformat(),
        "Ticker": "TOTAL",
        "Shares": "",
        "Buy Price": "",
        "Cost Basis": "",
        "Stop Loss": "",
        "Current Price": "",
        "Total Value": 0.0,
        "PnL": 0.0,
        "Action": "",
        "Cash Balance": float(args.starting_equity),
        "Total Equity": float(args.starting_equity),
    }

    _write_csv(daily_path, DAILY_UPDATES_HEADER, [total_row])
    _write_csv(trade_path, TRADE_LOG_HEADER, [])

    readme_path = exp_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                f"# Experimento: {args.name}",
                "",
                "Este experimento usa los mismos CSV schemas que el dataset histórico.",
                "",
                "Archivos:",
                f"- `{daily_path.as_posix()}`",
                f"- `{trade_path.as_posix()}`",
                "",
                "Ejecutar mantenimiento/registro (paper/live, según tu uso):",
                "",
                "```powershell",
                f".\\.venv311\\Scripts\\python.exe Experiments\\chatgpt_micro-cap\\scripts\\processing\\trading_script.py --data-dir \"{csv_dir.as_posix()}\" --starting-equity {args.starting_equity}",
                "```",
                "",
                "Recomendaciones (si usas este Daily Updates como universo/holdings):",
                "",
                "```powershell",
                f".\\.venv311\\Scripts\\python.exe .\\recommend.py --universe-from-daily --holdings-from-daily --daily-updates \"{daily_path.as_posix()}\"",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Experimento creado: {exp_dir}")
    print(f"- Daily Updates: {daily_path}")
    print(f"- Trade Log:     {trade_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

