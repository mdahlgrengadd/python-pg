"""
PGstandard.pl - Load standard PG macro packages.

Top-level barrel module for short imports (1:1 parity with Perl PGstandard.pl).
Loads: PG.pl, PGbasicmacros.pl, PGanswermacros.pl, PGauxiliaryFunctions.pl

Usage:
    from pg.standard import *
    DOCUMENT()
    TEXT("Problem text")

Note: This is a barrel module that provides a clean top-level import API.
It aggregates multiple submodules from pg.macros for user convenience.

Reference: macros/core/PGstandard.pl
"""

from pg.macros.PGstandard import *

__all__ = [
    # From PG.pl (pg_core)
    "PGEnvironment",
    "get_environment",
    "set_environment",
    "DOCUMENT",
    "ENDDOCUMENT",
    "TEXT",
    "BEGIN_TEXT",
    "END_TEXT",
    "HEADER_TEXT",
    "POST_HEADER_TEXT",
    "STOP_RENDERING",
    "ANS",
    "NAMED_ANS",
    "LABELED_ANS",
    "RECORD_ANS_NAME",
    "RECORD_IMPLICIT_ANS_NAME",
    "NEW_ANS_NAME",
    "ANS_NUM_TO_NAME",
    "RECORD_FORM_LABEL",
    "RECORD_EXTRA_ANSWERS",
    "ans_rule_count",
    "SOLUTION",
    "HINT",
    "COMMENT",
    "loadMacros",
    "install_problem_grader",
    "not_null",
    "DEBUG_MESSAGE",
    "WARN_MESSAGE",
    "random",
    "non_zero_random",
    "list_random",
    "random_coprime",
    "shuffle",
    "random_subset",
    "persistent_data",
    # From PGbasicmacros.pl (pg_basic_macros)
    "beginproblem",
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
    # From PGanswermacros.pl (pg_answer_macros)
    "num_cmp",
    "std_num_cmp",
    "fun_cmp",
    "str_cmp",
    "std_str_cmp",
    "interval_cmp",
    "vector_cmp",
    "matrix_cmp",
    # From PGauxiliaryFunctions.pl (pg_utilities)
    "gcf",
    "gcd",
    "lcm",
    "reduce_fraction",
    "sgn",
    "max_number",
    "min_number",
    "step",
    "fact",
    "C",
    "P",
    "lex_sort",
    "num_sort",
    "uniq",
]
