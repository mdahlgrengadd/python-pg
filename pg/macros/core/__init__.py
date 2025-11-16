"""
Core PG macros.
"""

# Import from pg.core (the main PG.pl port)
from .pg_core import (
    # Environment
    PGEnvironment,
    get_environment,
    set_environment,
    
    # Document lifecycle
    DOCUMENT,
    ENDDOCUMENT,
    _PG_init,
    
    # Text output
    TEXT,
    BEGIN_TEXT,
    END_TEXT,
    HEADER_TEXT,
    POST_HEADER_TEXT,
    STOP_RENDERING,
    
    # Answers
    ANS,
    NAMED_ANS,
    LABELED_ANS,
    RECORD_ANS_NAME,
    RECORD_IMPLICIT_ANS_NAME,
    NEW_ANS_NAME,
    ANS_NUM_TO_NAME,
    RECORD_FORM_LABEL,
    RECORD_EXTRA_ANSWERS,
    ans_rule_count,
    
    # Solution/Hint
    SOLUTION,
    HINT,
    COMMENT,
    
    # Macro loading
    loadMacros,
    
    # Grading
    install_problem_grader,
    
    # Utilities
    not_null,
    DEBUG_MESSAGE,
    WARN_MESSAGE,
    MODES,

    # Random
    random,
    non_zero_random,
    list_random,
    
    # Persistent data
    persistent_data,
)

# Keep legacy imports from pg.standard for compatibility
from .pg_standard import (
    image,
    bold,
    italic,
    underline,
    ans_rule,
    solution,
    hint,
    shuffle,
    random_subset,
)

# Import from pg.basic_macros
from .pg_basic_macros import (
    beginproblem,
)

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
    "MODES",

    # Random
    "random",
    "non_zero_random",
    "list_random",
    "shuffle",
    "random_subset",
    
    # Persistent data
    "persistent_data",
    
    # Legacy from pg.standard
    "image",
    "bold",
    "italic",
    "underline",
    "ans_rule",
    "solution",
    "hint",

    # From pg_basic_macros
    "beginproblem",
]
