"""
PGanswermacros.pl - Answer evaluator constructors

Python port of macros/core/PGanswermacros.pl
Provides num_cmp, fun_cmp, str_cmp, etc.

Reference: PGanswermacros.pl (lines 1-800)
"""

from typing import Any

from pg.answer import AnswerEvaluator
from pg.answer.evaluators.formula import FormulaEvaluator
from pg.answer.evaluators.interval import IntervalEvaluator
from pg.answer.evaluators.matrix import MatrixEvaluator
from pg.answer.evaluators.numeric import NumericEvaluator
from pg.answer.evaluators.string import StringEvaluator
from pg.answer.evaluators.vector import VectorEvaluator
from pg.math import Formula

# Export list
__exports__ = [
    "num_cmp",
    "fun_cmp",
    "str_cmp",
    "std_num_cmp",
    "std_str_cmp",
    "interval_cmp",
    "vector_cmp",
    "matrix_cmp",
]


def num_cmp(
    correct: float | str,
    *,
    tolType: str = "relative",
    tolerance: float = 0.001,
    zeroLevel: float = 1e-14,
    zeroLevelTol: float = 1e-12,
    **options: Any,
) -> AnswerEvaluator:
    """
    Create numeric answer evaluator.

    Args:
        correct: Correct answer (number or expression)
        tolType: Tolerance type ('relative', 'absolute', or 'sigfigs')
        tolerance: Tolerance value
        zeroLevel: Values below this are treated as zero
        zeroLevelTol: Tolerance for zero-level comparison
        **options: Additional evaluator options

    Returns:
        NumericEvaluator

    Example:
        >>> evaluator = num_cmp(42, tolerance=0.01)
        >>> result = evaluator.evaluate("42.005")
        >>> result.correct  # True

    Reference: PGanswermacros.pl::num_cmp
    """
    return NumericEvaluator(
        correct_answer=correct,
        tolerance=tolerance,
        tolerance_mode=tolType,
        zero_level=zeroLevel,
        zero_level_tolerance=zeroLevelTol,
        **options,
    )


def std_num_cmp(
    correct: float | str,
    **options: Any,
) -> AnswerEvaluator:
    """
    Standard numeric comparison (0.1% tolerance).

    Args:
        correct: Correct answer
        **options: Additional options

    Returns:
        NumericEvaluator with 0.1% tolerance

    Reference: PGanswermacros.pl::std_num_cmp
    """
    return num_cmp(correct, tolerance=0.001, **options)


def fun_cmp(
    correct: str | Formula,
    *,
    var: str | list[str] = "x",
    limits: list[tuple[float, float]] | None = None,
    numPoints: int = 5,
    tolType: str = "relative",
    tolerance: float = 0.001,
    **options: Any,
) -> AnswerEvaluator:
    """
    Create formula/function answer evaluator.

    Args:
        correct: Correct formula (string or Formula object)
        var: Variable name(s)
        limits: Test point limits for each variable
        numPoints: Number of test points
        tolType: Tolerance type
        tolerance: Tolerance value
        **options: Additional options

    Returns:
        FormulaEvaluator

    Example:
        >>> evaluator = fun_cmp("x^2 + 1", var="x")
        >>> result = evaluator.evaluate("x**2 + 1")
        >>> result.correct  # True

    Reference: PGanswermacros.pl::fun_cmp
    """
    # Normalize var to list
    if isinstance(var, str):
        var = [var]

    # Convert string to Formula if needed
    if isinstance(correct, str):
        correct = Formula(correct, variables=var)

    return FormulaEvaluator(
        correct_answer=correct,
        test_points=numPoints,
        limits=limits,
        tolerance=tolerance,
        tolerance_mode=tolType,
        **options,
    )


def str_cmp(
    correct: str,
    *,
    case_sensitive: bool = True,
    trim_whitespace: bool = True,
    mode: str = "exact",
    **options: Any,
) -> AnswerEvaluator:
    """
    Create string answer evaluator.

    Args:
        correct: Correct string
        case_sensitive: Whether comparison is case-sensitive
        trim_whitespace: Whether to trim whitespace
        mode: Comparison mode ('exact' or 'regex')
        **options: Additional options

    Returns:
        StringEvaluator

    Example:
        >>> evaluator = str_cmp("hello", case_sensitive=False)
        >>> result = evaluator.evaluate("HELLO")
        >>> result.correct  # True

    Reference: PGanswermacros.pl::str_cmp
    """
    return StringEvaluator(
        correct_answer=correct,
        case_sensitive=case_sensitive,
        trim_whitespace=trim_whitespace,
        mode=mode,
        **options,
    )


def std_str_cmp(correct: str, **options: Any) -> AnswerEvaluator:
    """
    Standard string comparison (case-insensitive).

    Args:
        correct: Correct string
        **options: Additional options

    Returns:
        StringEvaluator

    Reference: PGanswermacros.pl::std_str_cmp
    """
    return str_cmp(correct, case_sensitive=False, **options)


def interval_cmp(
    correct: str,
    *,
    tolerance: float = 0.001,
    tolType: str = "relative",
    **options: Any,
) -> AnswerEvaluator:
    """
    Create interval answer evaluator.

    Args:
        correct: Correct interval (e.g., "[1,5)")
        tolerance: Tolerance for endpoints
        tolType: Tolerance type
        **options: Additional options

    Returns:
        IntervalEvaluator

    Reference: PGanswermacros.pl (interval checking)
    """
    return IntervalEvaluator(
        correct_answer=correct,
        tolerance=tolerance,
        tolerance_mode=tolType,
        **options,
    )


def vector_cmp(
    correct: str | list[float],
    *,
    tolerance: float = 0.001,
    tolType: str = "relative",
    **options: Any,
) -> AnswerEvaluator:
    """
    Create vector answer evaluator.

    Args:
        correct: Correct vector
        tolerance: Tolerance for components
        tolType: Tolerance type
        **options: Additional options

    Returns:
        VectorEvaluator

    Reference: PGanswermacros.pl (vector checking)
    """
    return VectorEvaluator(
        correct_answer=correct,
        tolerance=tolerance,
        tolerance_mode=tolType,
        **options,
    )


def matrix_cmp(
    correct: str | list[list[float]],
    *,
    tolerance: float = 0.001,
    tolType: str = "relative",
    **options: Any,
) -> AnswerEvaluator:
    """
    Create matrix answer evaluator.

    Args:
        correct: Correct matrix
        tolerance: Tolerance for elements
        tolType: Tolerance type
        **options: Additional options

    Returns:
        MatrixEvaluator

    Reference: PGanswermacros.pl (matrix checking)
    """
    return MatrixEvaluator(
        correct_answer=correct,
        tolerance=tolerance,
        tolerance_mode=tolType,
        **options,
    )
