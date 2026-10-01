.DEFAULT_GOAL := help
PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help install lint format typecheck test check demo dashboard collect clean

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-11s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv and install the package with dev tools
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

lint: ## Run ruff lint and format checks
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format: ## Auto-format and fix lint issues
	$(BIN)/ruff format src tests
	$(BIN)/ruff check --fix src tests

typecheck: ## Run mypy
	$(BIN)/mypy

test: ## Run the test suite with coverage
	$(BIN)/pytest --cov --cov-report=term-missing

check: lint typecheck test ## Everything CI runs

demo: ## Run the full pipeline offline on synthetic data (data/demo)
	$(BIN)/obml demo --force

dashboard: ## Open the dashboard
	$(BIN)/obml dashboard

collect: ## Record 30 minutes of live Binance data
	$(BIN)/obml collect --minutes 30

clean: ## Remove caches and build output (keeps data/)
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
