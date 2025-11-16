"""
PGanswermacros.pl - Answer checking macros.

Top-level barrel module for short imports (1:1 parity with Perl PGanswermacros.pl).
Re-exports from pg.macros.answers.pg_answer_macros.

Usage:
    from pg.answermacros import num_cmp, fun_cmp
    ANS(num_cmp(42))

Reference: macros/answers/PGanswermacros.pl
"""

from pg.macros.answers.pg_answer_macros import (
    num_cmp,
    std_num_cmp,
    fun_cmp,
    str_cmp,
    std_str_cmp,
    interval_cmp,
    vector_cmp,
    matrix_cmp,
)

__all__ = [
    "num_cmp",
    "std_num_cmp",
    "fun_cmp",
    "str_cmp",
    "std_str_cmp",
    "interval_cmp",
    "vector_cmp",
    "matrix_cmp",
]
