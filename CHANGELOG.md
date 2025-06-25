# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Professional project structure with proper package organization
- Modern Python packaging with `pyproject.toml`
- Comprehensive testing framework with pytest
- Code quality tools (black, isort, flake8, mypy)
- Pre-commit hooks for automated code quality checks
- Continuous Integration with GitHub Actions
- Proper logging system with structured logging
- Custom exception classes for better error handling
- Configuration management with validation
- Development tools and Makefile for common tasks
- Contributing guidelines and documentation
- Type hints throughout the codebase

### Changed
- Improved code organization and structure
- Enhanced error handling and validation
- Better documentation and docstrings
- Modernized dependency management

### Fixed
- Code style inconsistencies
- Missing type hints
- Inadequate error handling
- Lack of testing infrastructure

## [1.0.0] - 2024-01-XX

### Added
- Initial release of ML-ADA
- 2D object detection support
- Confidence-based sampling strategy for active learning
- Kitti and OpenLabel data annotation formats
- Yolov3 for object detection
- API-based integration with CVAT
- GUI and command-line interfaces
- Model fine-tuning capabilities
- Annotation visualization tools

### Features
- Semi-automatic data annotation using ML models
- Active learning with model fine-tuning
- Support for multiple data formats
- Integration with external annotation tools
- Real-time model performance monitoring
- Configurable annotation workflows 