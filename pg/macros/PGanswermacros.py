"""
PGanswermacros.pl - Answer evaluation macros.

This module provides 1:1 parity with the Perl PGanswermacros.pl macro file.
Re-exports all answer comparison functions from pg.macros.answers.pg_answer_macros.

Reference: macros/core/PGanswermacros.pl
"""

from .answers.pg_answer_macros import (
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

