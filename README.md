# BCG Data Challenge

## Project Description

This project analyzes barley yield and climate data.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Prerequisites

Install `uv` if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

1. Install dependencies:
```bash
uv sync
```

2. Activate the virtual environment:
```bash
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

### Development Setup

Install development dependencies:

```bash
uv sync --extra dev
```

### Pre-commit Hooks

Install pre-commit hooks:

```bash
pre-commit install
```

The hooks will run automatically on commit. You can also run them manually:

```bash
pre-commit run --all-files
```

## Project Structure

```
.
├── src/              # Source code
├── tests/            # Test files
├── data/             # Data files
├── pyproject.toml    # Project configuration
└── README.md         # This file
```

## Running Tests

```bash
pytest
```

## Code Quality

This project uses:
- **black** for code formatting
- **ruff** for linting
- **mypy** for type checking
- **pytest** for testing

All checks run automatically via pre-commit hooks.
