# pg_macros

Python ports of WeBWorK PG macro library.

Provides compatibility layer for legacy .pg problems while maintaining modern Python architecture.

## Macro Categories

### Core (pg_macros.core)
- **PGstandard**: Basic PG functions (TEXT, ANS, image, etc.)
- **MathObjects**: MathObject integration
- **PGML**: PGML rendering support

### Answers (pg_macros.answers)
- **PGanswermacros**: Answer evaluators (num_cmp, fun_cmp, str_cmp, etc.)
- Answer formatting and checking utilities

### Parsers (pg_macros.parsers)
- **parserPopUp**: Popup menu parsers
- **parserRadioButtons**: Radio button parsers
- **parserCheckboxes**: Checkbox parsers

### Choice (pg_macros.choice)
- **PGchoicemacros**: Multiple choice, true/false, matching

## Usage

```python
from pg_macros import loadMacros

# Load macros into namespace
loadMacros("PGstandard.pl", "MathObjects.pl", "PGML.pl")

# Now use macro functions
TEXT("Problem statement")
ANS(num_cmp(42))
```

## Architecture

Uses registry pattern for dynamic macro loading:
1. Macros register themselves on import
2. `loadMacros()` loads from registry
3. Python implementations mirror Perl API

## Porting Guide

See `PORTING.md` for guide on porting Perl macros to Python.
