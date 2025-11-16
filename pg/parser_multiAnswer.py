"""
parserMultiAnswer.pl - Multi-answer checker.

Top-level barrel module for short imports (1:1 parity with Perl parserMultiAnswer.pl).
Re-exports from pg.macros.parsers.parserMultiAnswer.

Usage:
    from pg.parser_multiAnswer import MultiAnswer

Reference: macros/parsers/parserMultiAnswer.pl
"""

from pg.macros.parsers.parserMultiAnswer import *

__all__ = [
    "MultiAnswer",
]
