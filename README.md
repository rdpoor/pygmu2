# pygmu2

A Python project with modern tooling, logging, and testing infrastructure.

## Project Structure

```
pygmu2/
├── src/
│   └── pygmu2/
│       ├── __init__.py
│       ├── logger.py          # Logging configuration
│       └── __main__.py         # Entry point
├── tests/
│   └── test_pygmu2.py
├── Pipfile                     # pipenv dependencies
├── Pipfile.lock                # Locked dependencies
├── LICENSE                     # MIT License
├── README.md
└── .gitignore
```

## Prerequisites

Install pipenv if you haven't already:

```bash
pip install pipenv
```

## Installation

Install dependencies using pipenv:

```bash
pipenv install --dev
```

This will:
- Create a virtual environment
- Install all dependencies from Pipfile
- Install development dependencies

## Activating the Virtual Environment

Activate the pipenv shell:

```bash
pipenv shell
```

Or run commands within the virtual environment:

```bash
pipenv run python -m pytest
```

## Running Tests

Run all tests:
```bash
pipenv run pytest
# or use the script:
pipenv run test
```

Run tests with coverage:
```bash
pipenv run pytest --cov=src --cov-report=html
# or use the script:
pipenv run test-cov
```

## Logging

The project includes a logging module (`src/pygmu2/logger.py`) that provides:

- `setup_logging()` - Configure logging with custom levels and formats
- `get_logger()` - Get a logger instance for any module

Example usage:

```python
from pygmu2.logger import setup_logging, get_logger

# Set up logging (typically done once at application start)
logger = setup_logging(level="INFO")

# Or get a logger for a specific module
logger = get_logger(__name__)
logger.info("Application started")
```

## Development

### Code Formatting

Format code with Black:
```bash
pipenv run format
```

### Type Checking

Run type checking with mypy:
```bash
pipenv run typecheck
```

### Linting

Run linting with flake8:
```bash
pipenv run lint
```

## Adding Dependencies

Add a production dependency:
```bash
pipenv install <package-name>
```

Add a development dependency:
```bash
pipenv install --dev <package-name>
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
