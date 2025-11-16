"""
parserPopUp.pl - Pop-up menu parser.

Top-level barrel module for short imports (1:1 parity with Perl parserPopUp.pl).
Re-exports from pg.macros.parsers.parserPopUp.

Reference: macros/parsers/parserPopUp.pl
"""

from pg.macros.parsers.parserPopUp import *

__all__ = [
    "PopUp",
]
