# Defaults (Linux/macOS). Override with `make VENV_DIR=...` if you want.
VENV_DIR ?= venv
PY := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
DATA_DIR ?= Experiments/chatgpt_micro-cap/csv_files
MPLBACKEND ?= Agg

# Virtual environment setup
venv:
	python3 -m venv $(VENV_DIR)
	@echo "Virtual environment created. Activate with: source $(VENV_DIR)/bin/activate"

# Install dependencies
install: venv
	$(PIP) install -r requirements.txt

# Run the portfolio maintenance script (writes/reads CSVs under DATA_DIR)
trade: install
	$(PY) Experiments/chatgpt_micro-cap/scripts/processing/trading_script.py --data-dir "$(DATA_DIR)" $(ARGS)

# Generate a baseline graph from the CSVs (headless-friendly)
graph: install
	MPLBACKEND=$(MPLBACKEND) $(PY) Experiments/chatgpt_micro-cap/graphing/daily_returns.py

# Clean up virtual environment
clean:
	rm -rf $(VENV_DIR)
	@echo "Virtual environment removed: $(VENV_DIR)"

.PHONY: venv install trade graph clean

