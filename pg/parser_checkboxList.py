"""
parserCheckboxList.pl - Checkbox list parser.

Top-level barrel module for short imports (1:1 parity with Perl parserCheckboxList.pl).
Re-exports from pg.macros.parsers.parserCheckboxList.

Reference: macros/parsers/parserCheckboxList.pl
"""

from pg.macros.parsers.parserCheckboxList import *

__all__ = [
    "CheckboxList",
]
