"""
PGbasicmacros.pl - Basic PG macros for formatting and input.

Top-level barrel module for short imports (1:1 parity with Perl PGbasicmacros.pl).
Re-exports from pg.macros.core.pg_basic_macros.

Usage:
    from pg.basicmacros import ans_rule, PAR, BR
    ans_rule(10)
    PAR()

Reference: macros/core/PGbasicmacros.pl
"""

from pg.macros.core.pg_basic_macros import *

__all__ = [
    "_PGbasicmacros_init",
    "PAR",
    "BR",
    "BRBR",
    "LQ",
    "RQ",
    "BBOLD",
    "EBOLD",
    "BITALIC",
    "EITALIC",
    "BUL",
    "EUL",
    "BCENTER",
    "ECENTER",
    "HR",
    "NBSP",
    "PI",
    "E",
    "ans_rule",
    "NAMED_ANS_RULE",
    "ans_box",
    "NAMED_ANS_BOX",
    "ans_radio_buttons",
    "NAMED_ANS_RADIO_BUTTONS",
    "pop_up_list",
    "NAMED_POP_UP_LIST",
    "MODES",
    "image",
]
