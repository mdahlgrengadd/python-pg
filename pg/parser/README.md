# pg_parser - Mathematical Expression Parser for WeBWorK PG

**Status**: Phase 1 Complete ✅

A modern Python implementation of the mathematical expression parser for the WeBWorK Problem Generator (PG) system.

## Features

- ✅ **Tokenization**: Regex-based tokenization with context-aware patterns
- ✅ **AST Construction**: Full Abstract Syntax Tree representation
- ✅ **Recursive Descent Parser**: Operator precedence (Pratt parsing) with context-driven rules
- ✅ **Context System**: Mathematical environments defining variables, constants, functions, operators
- ✅ **AST Visitors**: String rendering, LaTeX rendering, evaluation
- ✅ **Implicit Multiplication**: Automatic insertion (e.g., `2x` → `2*x`)
- ✅ **Multiple Bracket Types**: Parentheses `()`, brackets `[]`, angle brackets `<>`, pipes `||`

## Installation

Install the unified `pg` package from the repo root:

```bash
pip install -e packages
```

## Usage

### Basic Parsing

```python
from pg_parser import Parser

parser = Parser()
ast = parser.parse("2*x^2 + 3*x + 1")
print(ast)  # AST representation
```

### String Rendering

```python
from pg_parser import Parser
from pg_parser.visitors import StringVisitor

parser = Parser()
ast = parser.parse("2*x + 3")

visitor = StringVisitor()
result = ast.accept(visitor)
print(result)  # "2 * x + 3"
```

### LaTeX Rendering

```python
from pg_parser import Parser
from pg_parser.visitors import TeXVisitor

parser = Parser()
ast = parser.parse("x^2 / 2")

visitor = TeXVisitor()
result = ast.accept(visitor)
print(result)  # "\\frac{x^{2}}{2}"
```

### Evaluation

```python
from pg_parser import Parser
from pg_parser.visitors import EvalVisitor

parser = Parser()
ast = parser.parse("2*x + 3")

visitor = EvalVisitor(bindings={"x": 5.0})
result = ast.accept(visitor)
print(result)  # 13.0
```

### Custom Contexts

```python
from pg_parser import Parser, Context

# Use built-in contexts
context = Context.complex()  # Includes imaginary unit 'i'
parser = Parser(context)
ast = parser.parse("2 + 3*i")

# Or create custom contexts
context = Context.numeric()
context.variables["t"] = VariableConfig(name="t")
context.functions["f"] = FunctionConfig(name="f", min_args=1, max_args=1)
```

## Architecture

### Components

1. **Tokenizer** (`tokenizer.py`)
   - Regex-based pattern matching
   - Context-aware tokenization
   - Implicit multiplication insertion
   - Position tracking for error reporting

2. **AST Nodes** (`ast.py`)
   - Hierarchical node types: Number, Variable, Constant, String
   - Binary/Unary operators
   - Function calls
   - Collections: List, Point, Vector, Matrix, Interval
   - Visitor pattern for extensibility

3. **Parser** (`parser.py`)
   - Recursive descent with operator precedence (Pratt parsing)
   - Context-driven parsing rules
   - Error recovery and reporting

4. **Context System** (`context.py`)
   - Mathematical environment definitions
   - Operator precedence and associativity
   - Variable, constant, and function declarations
   - Tolerance settings for fuzzy comparison
   - Built-in contexts: Numeric, Complex, Vector, Interval

5. **Visitors** (`visitors.py`)
   - **StringVisitor**: AST → string representation
   - **TeXVisitor**: AST → LaTeX
   - **EvalVisitor**: AST → numeric evaluation

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pg_parser --cov-report=term-missing

# Current status: 42 tests, 67% coverage
```

## Examples

### Implicit Multiplication

```python
parser.parse("2x")           # 2 * x
parser.parse("2(x+1)")       # 2 * (x+1)
parser.parse("(x+1)(x-1)")   # (x+1) * (x-1)
parser.parse("sin x")        # sin(x) - recognized as function
```

### Operator Precedence

```python
parser.parse("2 + 3 * 4")    # 2 + (3 * 4)
parser.parse("2^3^4")        # 2^(3^4) - right associative
parser.parse("(2 + 3) * 4")  # Explicit grouping
```

### Mathematical Structures

```python
parser.parse("(1, 2, 3)")      # Point with 3 coordinates
parser.parse("<1, 2, 3>")      # Vector with 3 components
parser.parse("[1, 2, 3]")      # List
parser.parse("[[1,2],[3,4]]")  # 2x2 Matrix
parser.parse("[0, 1)")         # Half-open interval
```

### Functions

```python
parser.parse("sin(x)")              # Single argument
parser.parse("max(1, 2, 3)")        # Multiple arguments
parser.parse("sqrt(2)")             # Built-in functions
parser.parse("|x|")                 # Absolute value (pipes)
```

## Comparison with Legacy Perl System

This implementation provides equivalent functionality to:
- `lib/Parser.pm` - Tokenization and parsing
- `lib/Parser/Context.pm` - Context system
- `lib/Parser/*.pm` - AST node types

### Improvements

1. **Type Safety**: Python type hints with mypy strict mode
2. **Modern Patterns**: Visitor pattern for extensibility
3. **Data-Driven**: Contexts defined via YAML (planned)
4. **Testability**: 42 unit tests with property-based testing support
5. **Performance**: Compiled regex, efficient AST traversal

## Next Steps (Phase 2)

The next phase will implement the MathObjects system (equivalent to `lib/Value.pm`):

- Intelligent mathematical value types (Real, Complex, Vector, Matrix, etc.)
- Operator overloading with type promotion
- Fuzzy comparison with configurable tolerances
- Formula type for deferred evaluation

## Reference

This implementation is based on the legacy WeBWorK PG system:
- **Parser**: `lib/Parser.pm` (lines 100-171)
- **Context**: `lib/Parser/Context.pm`
- **AST Nodes**: `lib/Parser/*.pm` subdirectories

## License

Same as WeBWorK PG system (GPLv2+)
