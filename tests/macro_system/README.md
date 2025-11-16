# Macro System Tests

This directory contains tests for the Python port of the PG macro system.

## Test Files

### `test_pg_core.py`
Tests core PG.pl functionality:
- Environment creation and management
- Document lifecycle (DOCUMENT/ENDDOCUMENT)
- Text accumulation (TEXT, HEADER_TEXT)
- Answer registration (ANS, NAMED_ANS)
- Answer name generation
- Random number functions
- Utility functions
- Persistent data storage
- Solution/Hint flags

**Run:** `PYTHONPATH=packages/pg_macros python3 tests/macro_system/test_pg_core.py`

### `test_pg_basic_macros.py`
Tests PGbasicmacros.pl functionality:
- Display mode constants (PAR, BR, BBOLD, etc.)
- Answer blanks (ans_rule, ans_box)
- Radio buttons (ans_radio_buttons)
- Dropdown lists (pop_up_list)
- MODES() function
- Image insertion
- Named answer blanks

**Run:** `PYTHONPATH=packages/pg_macros python3 tests/macro_system/test_pg_basic_macros.py`

### `test_pg_problem_complete.py`
End-to-end tests with complete PG problems:
- Hello World problem
- Multi-answer problems
- Named answer problems  
- Problems with solutions and hints

**Run:** `PYTHONPATH=packages/pg_macros python3 tests/macro_system/test_pg_problem_complete.py`

## Test Coverage

- **pg_core.py**: 10 tests - All passing ✅
- **pg_basic_macros.py**: 9 tests - All passing ✅
- **Complete problems**: 4 tests - All passing ✅

**Total: 23 tests, 100% passing**

## Example Usage

```bash
# Run all tests
cd /home/runner/work/pg/pg
for test in tests/macro_system/test_*.py; do
    echo "Running $test..."
    PYTHONPATH=packages/pg_macros python3 "$test"
done
```
