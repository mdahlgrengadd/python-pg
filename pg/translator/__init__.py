"""
pg_translator - PG problem file translator and executor

Executes .pg problem files safely, collecting problem text, answers, solutions, and hints.

Reference: lib/WeBWorK/PG/Translator.pm in legacy Perl codebase
"""

# ============================================================================
# IMPORTANT: PGPreprocessor is imported from pg.preprocessor_pygment.py
# ============================================================================
# The Lark grammar-based preprocessor (PGPreprocessor) is the ONLY supported
# preprocessor. It properly parses PG/Perl syntax and transforms it to Python.
#
# The old regex-based preprocessor (LegacyPGPreprocessor) is DEPRECATED and
# has known bugs. DO NOT USE IT except for compatibility testing.
#
# ALWAYS use PGPreprocessor from pg.preprocessor_pygment.py
# ============================================================================

from .pg_preprocessor_pygment import PGPreprocessor, convert_pg_file
from .preprocessor import PGPreprocessor as LegacyPGPreprocessor  # DEPRECATED
from .executor import PGExecutor, PGEnvironment
from .translator import PGTranslator, ProblemResult

__all__ = [
    "PGPreprocessor",         # The CORRECT preprocessor (Lark-based)
    "LegacyPGPreprocessor",   # DEPRECATED - DO NOT USE
    "convert_pg_file",
    "PGExecutor",
    "PGEnvironment",
    "PGTranslator",
    "ProblemResult",
]
