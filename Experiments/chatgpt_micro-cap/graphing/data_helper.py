import pandas as pd 
import os
from datetime import date
from pathlib import Path

overall_dir = Path(__file__).parents[1]
repo_root_dir = Path(__file__).parents[3]
TRADE_LOG_PATH = overall_dir / Path("csv_files/Trade Log.csv")
DAILY_PATH = overall_dir / Path("csv_files/Daily Updates.csv")

def load_data(trade_log_path: str | Path = TRADE_LOG_PATH, daily_updates_path: str | Path = DAILY_PATH):
    trades = pd.read_csv(trade_log_path, parse_dates=["Date"])
    daily = pd.read_csv(daily_updates_path, parse_dates=["Date"])
    equity = daily[daily["Ticker"] == "TOTAL"].sort_values("Date")
    return trades, daily, equity

def _default_plots_dir() -> Path:
    # Align with the rest of the repo: write generated artifacts under runs/ (gitignored).
    asof = os.getenv("ASOF_DATE")
    date_str = asof if asof else date.today().isoformat()
    return repo_root_dir / "runs" / date_str / "plots"

def assemble_path(file_name: str) -> Path:
    override = os.getenv("LLM_TRADING_LAB_PLOTS_DIR")
    plots_dir = Path(override) if override else _default_plots_dir()
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir / file_name



