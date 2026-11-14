# SentinelPay — run `make help` for targets.
SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python3.11
VENV   := .venv
BIN    := $(VENV)/bin
CLI    := $(BIN)/sentinelpay

# Spark 4 needs Java 17 or 21. Resolve it per recipe (fails with install advice).
WITH_JAVA := JAVA_HOME=$$(./scripts/find_java.sh) && export JAVA_HOME &&

.PHONY: help install demo generate bronze silver gold train promote mlflow docs test test-fast lint format lock clean \
        app airflow mlflow demo-artifacts

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(VENV)/.installed: pyproject.toml requirements.lock
	@command -v $(PYTHON) >/dev/null || { echo "ERROR: $(PYTHON) not found (Python 3.11 required)"; exit 1; }
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -r requirements.lock
	$(BIN)/pip install -q -e . --no-deps
	@touch $@

install: $(VENV)/.installed ## Create .venv with pinned dependencies

demo: install ## Run the whole pipeline end to end (SAMPLE_SIZE=... to resize)
	@$(WITH_JAVA) $(CLI) demo

generate: install ## Generate synthetic history + production data
	@$(CLI) generate

bronze: install ## Ingest landing data into the bronze Delta table
	@$(WITH_JAVA) $(CLI) bronze

silver: install ## Build silver + quarantine tables and DQ metrics
	@$(WITH_JAVA) $(CLI) silver

gold: install ## Build the point-in-time gold feature table
	@$(WITH_JAVA) $(CLI) gold

train: install ## Train + evaluate models, register a challenger in MLflow
	@$(CLI) train

promote: install ## Promote the challenger to champion if it is better
	@$(CLI) promote

mlflow: install ## Open the MLflow UI at http://localhost:5000
	@MLFLOW_DISABLE_AGENT_HINT=1 $(BIN)/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docs: install ## Regenerate docs/features.md from the feature registry
	@$(CLI) features-doc

test: install ## Run all tests with coverage
	@$(WITH_JAVA) $(BIN)/pytest --cov --cov-report=term-missing

test-fast: install ## Run tests that don't need Spark
	@$(BIN)/pytest -m "not spark"

lint: install ## Lint and check formatting
	@$(BIN)/ruff check src tests app
	@$(BIN)/ruff format --check src tests app

format: install ## Auto-fix lint issues and format code
	@$(BIN)/ruff check --fix src tests app
	@$(BIN)/ruff format src tests app

lock: ## Re-resolve requirements.lock from pyproject.toml (maintainers only)
	$(PYTHON) -m venv /tmp/sentinelpay-lock && /tmp/sentinelpay-lock/bin/pip install -q -e ".[dev]"
	/tmp/sentinelpay-lock/bin/pip check  # never write an inconsistent lock file
	/tmp/sentinelpay-lock/bin/pip freeze --exclude-editable > requirements.lock
	rm -rf /tmp/sentinelpay-lock

clean: ## Delete generated data and reports
	rm -rf data reports mlruns mlflow.db spark-warehouse metastore_db derby.log

app: install ## Launch the dashboard at http://localhost:8501 (run `make demo` first)
	@$(BIN)/streamlit run app/streamlit_app.py

airflow demo-artifacts:
	@echo "'make $@' arrives in a later phase (see the roadmap in README.md)."
