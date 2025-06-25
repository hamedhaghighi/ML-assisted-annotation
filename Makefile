# Makefile for Python code quality

.PHONY: lint format isort type-check check clean

# Lint with flake8 or fallback to pylint (cross-platform: works on Windows and Unix)
lint:
	flake8 .

# Format code with black
format:
	@echo "Formatting code with black..."
	@black .

# Sort imports with isort
isort:
	@echo "Sorting imports with isort..."
	@isort .

# Type check with mypy
type-check:
	@echo "Type checking with mypy..."
	@mypy .

# Run all code quality checks
check: lint type-check

# Clean Python cache and build artifacts
clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete 