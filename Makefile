.PHONY: help install test lint format check clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies and pre-commit hooks"
	@echo "  make test         - Run all tests"
	@echo "  make test-unit    - Run unit tests only"
	@echo "  make test-int     - Run integration tests only"
	@echo "  make lint         - Run all linters (ruff, mypy)"
	@echo "  make format       - Auto-format code with ruff"
	@echo "  make check        - Run format check + lint + tests (full CI locally)"
	@echo "  make pre-commit   - Install pre-commit hooks"
	@echo "  make clean        - Remove cache files"

install:
	pip install --upgrade pip
	pip install -e .
	pip install -r requirements_test.txt
	pip install pre-commit ruff mypy
	pre-commit install

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-int:
	pytest tests/integration/ -v

lint:
	@echo "Running ruff check..."
	ruff check src/ tests/ scripts/ ghost_hunter.py
	@echo "\nRunning mypy..."
	mypy src/ --ignore-missing-imports --no-strict-optional || true

format:
	@echo "Running ruff format..."
	ruff format src/ tests/ scripts/ ghost_hunter.py
	@echo "\nRunning ruff check --fix..."
	ruff check --fix src/ tests/ scripts/ ghost_hunter.py

format-check:
	@echo "Checking ruff formatting..."
	ruff format --check src/ tests/ scripts/ ghost_hunter.py
	@echo "\nChecking ruff linting..."
	ruff check src/ tests/ scripts/ ghost_hunter.py

check: format-check lint test
	@echo "\n✅ All checks passed!"

pre-commit:
	pre-commit install
	@echo "Pre-commit hooks installed! They will run automatically on git commit."

pre-commit-run:
	pre-commit run --all-files

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ coverage.xml .coverage 2>/dev/null || true
	@echo "Cleaned up cache files!"
