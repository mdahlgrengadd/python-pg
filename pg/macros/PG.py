"""
PG.pl - Core Program Generation Language functionality.

This module provides 1:1 parity with the Perl PG.pl macro file.
Re-exports all functions from pg.macros.core.pg_core.

Reference: macros/PG.pl
"""

from .core.pg_core import *

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
