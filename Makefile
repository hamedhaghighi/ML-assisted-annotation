.PHONY: help install install-dev test test-cov lint format clean build docs

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in development mode
	pip install -e .

install-dev:  ## Install the package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=. --cov-report=html --cov-report=term-missing

lint:  ## Run linting tools
	flake8 .
	mypy .
	black --check .
	isort --check-only .

format:  ## Format code
	black .
	isort .

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:  ## Build the package
	python -m build

docs:  ## Build documentation
	# Add documentation building commands here when docs are added
	@echo "Documentation building not yet implemented"

check: lint test  ## Run all checks (lint + test)

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

docker-build:  ## Build Docker image
	docker build -f docker/Dockerfile-dev -t ml-ada:latest .

docker-run:  ## Run Docker container
	docker run -it --rm -v $(PWD):/workspace ml-ada:latest

setup-dev: install-dev  ## Setup development environment
	@echo "Development environment setup complete!"
	@echo "Run 'make help' to see available commands" 