# pg_pgml

PGML (PG Markup Language) parser and renderer for WeBWorK.

PGML is a markdown-like language for authoring WeBWorK problems with:
- Variable interpolation: `[$var]`
- Answer blanks: `[_____]`, `[@ ans_rule() @]`
- Code execution: `[@ code @]`
- Math display: `[```math```]`, `[``latex``]`

## Features

- Tokenizer for PGML syntax
- Document parser (blocks and inline elements)
- HTML renderer
- TeX renderer
- Variable interpolation
- Answer blank handling

## Reference

Based on `macros/core/PGML.pl` from legacy Perl codebase.
