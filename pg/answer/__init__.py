"""
pg_answer - Answer evaluation framework for WeBWorK PG problems

Provides pluggable answer checking system with:
- Type-specific evaluators
- Fuzzy comparison with tolerances
- Partial credit support
- Custom grading strategies

Reference: lib/AnswerHash.pm and lib/AnswerEvaluator.pm in legacy Perl codebase
"""

from .answer_hash import AnswerResult
from .cmp import checkbox_cmp, fun_cmp, num_cmp, radio_cmp, str_cmp
from .evaluator import AnswerEvaluator, EvaluatorRegistry
from .graders import AverageGrader, Grader, StandardGrader

__all__ = [
    "AnswerResult",
    "AnswerEvaluator",
    "EvaluatorRegistry",
    "Grader",
    "StandardGrader",
    "AverageGrader",
    # Convenience functions
    "num_cmp",
    "str_cmp",
    "fun_cmp",
    "radio_cmp",
    "checkbox_cmp",
]
