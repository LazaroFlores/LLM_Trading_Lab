"""Wrapper for the shared trading script using local data directory."""

from pathlib import Path
import sys

# Allow importing the trading script from this folder
sys.path.append(str(Path(__file__).resolve().parent))
from trading_script import main


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parents[2] / "csv_files"
    main(data_dir)


