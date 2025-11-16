# Python PG Library

A Python implementation of the WeBWorK Problem Generation (PG) system for creating and rendering mathematical problem sets.

## Overview

This repository contains a standalone Python implementation of the PG system, extracted from the main monorepo. It provides:

- **PG Problem Translation**: Convert `.pg` problem files to rendered HTML
- **Answer Checking**: Validate student answers against correct solutions
- **Math Rendering**: Support for LaTeX math notation
- **Macro System**: Extensive library of PG macros for problem creation
- **Tutorial Problems**: Sample problems demonstrating various features

## Installation

### Requirements

- Python 3.10 or higher
- pip

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd python-pg

# Install in development mode (recommended for development)
# This includes all dependencies including dev tools
pip install -e ".[dev]"

# Or install without dev dependencies (for production use)
pip install -e .
```

**Note**: For development, it's recommended to install with `[dev]` to get testing and linting tools.

### Dependencies

The package requires:
- `lark>=1.1.0` - Parser generator for PG code preprocessing
- `numpy>=1.24` - Numerical computations
- `pygments>=2.15.0` - Syntax highlighting for code preprocessing
- `pyyaml>=6.0` - YAML configuration parsing
- `RestrictedPython>=6.0` - Safe code execution sandbox
- `sympy>=1.12` - Symbolic mathematics

## Usage

### Command-Line Tools

#### `pg_solve.py` - Interactive Problem Solver

Solve PG problems interactively:

```bash
python pg_solve.py tutorial/sample-problems/Algebra/ExpandedPolynomial.pg --seed 1234
```

Options:
- `--seed N`: Set random seed for problem generation
- `--solution`: Show solution after answering
- `--hint`: Show hint if available
- `--no-check`: Skip answer checking (just display problem)

#### Legacy Tools (Deprecated)

**Note**: The `pypg.py` utility script has been removed as it was a legacy tool
using the deprecated `pg/renderer` pathway. For testing problems, use `pg_solve.py`
instead, which provides full Perl syntax support and accurate rendering.

### Python API

```python
from pg.translator import PGTranslator

# Create translator
translator = PGTranslator()

# Translate a problem file
result = translator.translate("path/to/problem.pg", seed=1234)

# Access rendered content
print(result.statement_html)
print(result.answer_blanks)

# Check answers
result = translator.translate("path/to/problem.pg", seed=1234, inputs={"ANS1": "x^2+1"})
print(result.answer_results)
```

## Project Structure

```
python-pg/
├── pg/                    # Main package
│   ├── answer/            # Answer checking and evaluation
│   ├── math/              # Mathematical objects and operations
│   ├── parser/            # PG code parsing
│   ├── pgml/              # PGML (PG Markup Language) support
│   ├── macros/            # PG macro library
│   ├── translator/        # Problem translation engine
│   └── renderer/          # HTML rendering
├── tests/                 # Test suite
├── tutorial/              # Tutorial problems
│   └── sample-problems/   # Example problems
├── pg_solve.py           # Interactive problem solver
└── pyproject.toml        # Package configuration
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pg --cov-report=html

# Run specific test file
pytest tests/translator/test_tutorial_sample_problems_all.py
```

## Tutorial Problems

The `tutorial/sample-problems/` directory contains example problems organized by topic:

- **Algebra**: Polynomials, factoring, equations
- **Calculus**: Derivatives, integrals, limits
- **Linear Algebra**: Matrices, vectors
- **Statistics**: Data analysis, probability
- **And more...**

Each problem demonstrates different PG features and answer types.

## Development

### Code Style

The project uses:
- **Black** for code formatting (88 column limit)
- **Ruff** for linting
- **mypy** for type checking

Format and lint:

```bash
ruff check .
black .
mypy pg
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

See [LICENSE](LICENSE) file for details.

## Related Projects

This repository is extracted from the main monorepo. For the full project including backend services and web UI, see the original repository.

## Documentation

- [PG Documentation](https://webwork.maa.org/wiki/Category:PG)
- [WeBWorK Wiki](https://webwork.maa.org/wiki/Main_Page)

## Support

For issues and questions, please open an issue on GitHub.
