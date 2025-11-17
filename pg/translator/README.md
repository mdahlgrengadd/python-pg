# pg_translator

PG problem file translator and executor for WeBWorK.

Executes .pg problem files safely, collecting problem text, answers, solutions, and hints.

## Features

- PG file preprocessing (BEGIN_TEXT/BEGIN_PGML expansion)
- Safe code execution with restricted sandbox
- Problem text and answer collection
- Macro loading framework
- Error reporting with line numbers

## Security Model

The translator uses an in-process sandbox with custom restricted builtins:

### InProcessSandbox

**How it works:**
- Replaces Python's standard `__builtins__` with a whitelist of safe functions
- Restricts module imports via a `safe_import()` function
- Removes dangerous functions: `eval`, `exec`, `compile`, `open`, `__import__` (without whitelist)
- Allows only whitelisted modules: `pg.*`, `math`, `random`, `re`, `pydantic`

**Whitelisted builtins:**
- Type constructors: `int`, `float`, `str`, `bool`, `list`, `tuple`, `dict`, `set`, `frozenset`
- Math functions: `abs`, `round`, `pow`, `min`, `max`, `sum`
- Utilities: `len`, `range`, `enumerate`, `zip`, `map`, `filter`, `sorted`, `reversed`, `all`, `any`, `iter`, `next`
- String functions: `chr`, `ord`, `repr`, `format`
- Type checking: `isinstance`, `issubclass`, `type`, `hasattr`, `getattr`, `setattr`, `delattr`
- Decorators: `property`, `classmethod`, `staticmethod`, `super`
- Class creation: `__build_class__`
- Exception classes: `Exception`, `ValueError`, `TypeError`, `KeyError`, `IndexError`, `AttributeError`, `ZeroDivisionError`

**Whitelisted module imports:**
- `pg.*` - All python-pg modules
- `pydantic` - Data validation library (safe - no file/network/subprocess access)
- `math`, `random`, `re` - Standard library utilities

**What's blocked:**
- File system access: `open`, `pathlib`, `os.path`, etc.
- Network access: `socket`, `urllib`, `requests`, etc.
- Subprocess execution: `subprocess`, `os.system`, etc.
- Code execution: `eval`, `exec`, `compile` (without prior whitelist)
- Arbitrary module imports: only whitelisted modules allowed

### Why Pydantic is Safe

Pydantic (for BaseModel, Field, validators, ConfigDict) is allowed because:
- It's a data validation library with no dangerous capabilities
- It only imports stdlib modules internally (`typing`, `dataclasses`, `datetime`, etc.)
- It cannot access files, networks, or spawn subprocesses
- It's already tested extensively in this codebase (94+ passing tests)

### Subprocess Sandbox (Legacy)

**Note:** A subprocess-based sandbox exists in `sandbox.py` but is NOT recommended for production use.

**Security limitations:**
- No import restrictions - subprocess can import ANY module
- Can access the filesystem
- Can spawn other subprocesses
- Process isolation is the only security mechanism

## Architecture

The translator pipeline:

1. **Preprocessing** (`pg_preprocessor_pygment.py`)
   - Converts `.pg` file to `.pyg` (Python) file
   - Expands Perl macros to Python imports
   - Preserves BEGIN_TEXT/BEGIN_PGML blocks

2. **Execution** (`in_process_sandbox.py`)
   - Executes preprocessed Python code in restricted sandbox
   - Collects output text, answers, solutions, hints
   - Returns structured `ExecutionResult`

3. **Problem Context** (`pg_core.py`)
   - Provides PG functions: TEXT(), ANS(), SOLUTION(), HINT(), etc.
   - Manages PGEnvironment for state
   - Handles answer registration

4. **Math System** (`pg/math/`)
   - Math value types: Real, Complex, Vector, Matrix, Formula, Interval, Fraction
   - Context system for parsing and evaluation
   - Answer checking and grading

5. **Macro System** (`pg/macros/`)
   - Ports of WeBWorK PG macros
   - Parsers, UI elements, graders, utilities
   - Dynamically loaded based on `loadMacros()` calls

## Reference

Based on `lib/WeBWorK/PG/Translator.pm` from legacy Perl codebase.
