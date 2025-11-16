"""
PGstandard.pl - Load standard PG macro packages.

This module provides 1:1 parity with the Perl PGstandard.pl macro file.
PGstandard.pl in Perl loads: PG.pl, PGbasicmacros.pl, PGanswermacros.pl, PGauxiliaryFunctions.pl

Reference: macros/core/PGstandard.pl
"""

# Import from PG.pl equivalent
from .core.pg_core import *
# Import from PGbasicmacros.pl equivalent
from .core.pg_basic_macros import *
# Import from PGanswermacros.pl equivalent
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
# Import from PGauxiliaryFunctions.pl equivalent
from .core.pg_utilities import (
    gcf,
    gcd,
    lcm,
    reduce_fraction,
    sgn,
    max_number,
    min_number,
    step,
    fact,
    C,
    P,
    lex_sort,
    num_sort,
    uniq,
)
# Import additional standard functions
from .core.pg_standard import (
    shuffle,
    random_subset,
)

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
