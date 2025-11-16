# pg_translator

PG problem file translator and executor for WeBWorK.

Executes .pg problem files safely, collecting problem text, answers, solutions, and hints.

## Features

- PG file preprocessing (BEGIN_TEXT/BEGIN_PGML expansion)
- Safe code execution with RestrictedPython
- Problem text and answer collection
- Macro loading framework
- Error reporting with line numbers

## Reference

Based on `lib/WeBWorK/PG/Translator.pm` from legacy Perl codebase.
