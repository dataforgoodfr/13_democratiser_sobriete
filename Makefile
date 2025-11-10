.PHONY: help install dev test lint format clean run deploy

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make dev        - Install dev dependencies"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code"
	@echo "  make clean      - Clean cache files"

install:
	uv sync

dev:
	uv pip install -e ".[dev]"

test:
	uv run pytest tests/ -v

test-cov:
	pytest --cov=src/bia_backend --cov-report=html

lint:
	uv tool run ruff check src/ tests/
	uv tool run mypy src/

format:
	uv tool run ruff format src/ tests/ rag_system/pipeline_scripts
	uv tool run isort src/ tests/ rag_system/pipeline_scripts


clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache