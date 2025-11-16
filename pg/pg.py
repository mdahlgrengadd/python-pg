"""
PG.pl - Core Program Generation Language functionality.

Top-level barrel module for short imports (1:1 parity with Perl PG.pl).
Re-exports from pg.macros.PG.

Usage:
    import pg.pg
    pg.pg.DOCUMENT()
    pg.pg.TEXT("Problem text")
    pg.pg.ANS(answer)

Or use as a top-level import:
    from pg import PG
    PG.DOCUMENT()

Reference: macros/PG.pl
"""

from pg.macros.PG import *

__all__ = [
    # Environment
    "PGEnvironment",
    "get_environment",
    "set_environment",
    # Document lifecycle
    "DOCUMENT",
    "ENDDOCUMENT",
    "_PG_init",
    # Text output
    "TEXT",
    "BEGIN_TEXT",
    "END_TEXT",
    "HEADER_TEXT",
    "POST_HEADER_TEXT",
    "STOP_RENDERING",
    # Answers
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
    # Solution/Hint
    "SOLUTION",
    "HINT",
    "COMMENT",
    # Macro loading
    "loadMacros",
    # Grading
    "install_problem_grader",
    # Utilities
    "not_null",
    "DEBUG_MESSAGE",
    "WARN_MESSAGE",
    # Random
    "random",
    "non_zero_random",
    "list_random",
    # Persistent data
    "persistent_data",
]
