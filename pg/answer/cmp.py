"""
Answer checker convenience functions (PG compatibility layer).

These functions provide PG-style answer checker creation, matching the
original Perl interface from PGanswermacros.pl.

Reference: macros/PGanswermacros.pl
"""

from __future__ import annotations

from typing import Any

from pg.math import Formula
from pg.parser import Parser

from .evaluators import FormulaEvaluator, NumericEvaluator, StringEvaluator


def num_cmp(
    correct_answer: float | int | str,
    mode: str = "std",
    tolerance: float = 0.001,
    tolerance_mode: str = "relative",
    absolute_tolerance: float | None = None,
    relative_tolerance: float | None = None,
    zero_level: float = 1e-14,
    zero_level_tolerance: float = 1e-12,
    **options: Any,
) -> NumericEvaluator:
    """
    Create numeric answer checker.

    Args:
        correct_answer: Correct numerical answer
        mode: Comparison mode (default "std")
        tolerance: Tolerance for comparison (default 0.001 = 0.1%)
        tolerance_mode: "relative", "absolute", or "sigfigs"
        absolute_tolerance: Override absolute tolerance
        relative_tolerance: Override relative tolerance
        zero_level: Values below this treated as zero
        zero_level_tolerance: Tolerance when near zero
        **options: Additional evaluator options

    Returns:
        NumericEvaluator configured with specified options

    Examples:
        >>> ANS(num_cmp(42))
        >>> ANS(num_cmp(3.14, tolerance=0.01))
        >>> ANS(num_cmp(100, tolerance=0.5, tolerance_mode="absolute"))

    Reference:
        PGanswermacros.pl::num_cmp (lines 400-800)
    """
    # Handle tolerance overrides
    if relative_tolerance is not None:
        tolerance = relative_tolerance
        tolerance_mode = "relative"
    elif absolute_tolerance is not None:
        tolerance = absolute_tolerance
        tolerance_mode = "absolute"

    return NumericEvaluator(
        correct_answer=correct_answer,
        tolerance=tolerance,
        tolerance_mode=tolerance_mode,
        **options,
    )


def str_cmp(
    correct_answer: str,
    mode: str = "std",
    case_sensitive: bool = False,
    trim_whitespace: bool = True,
    regex_match: bool = False,
    filters: list[str] | None = None,
    **options: Any,
) -> StringEvaluator:
    """
    Create string answer checker.

    Args:
        correct_answer: Correct string answer
        mode: Comparison mode (default "std")
        case_sensitive: Whether comparison is case-sensitive (default False)
        trim_whitespace: Trim leading/trailing whitespace (default True)
        regex_match: Treat answer as regex pattern (default False)
        filters: Text filters to apply (not yet implemented)
        **options: Additional evaluator options

    Returns:
        StringEvaluator configured with specified options

    Examples:
        >>> ANS(str_cmp("Paris"))  # Case-insensitive
        >>> ANS(str_cmp("Hello", case_sensitive=True))
        >>> ANS(str_cmp(r"\\d+", regex_match=True))

    Reference:
        PGanswermacros.pl::str_cmp (lines 1200-1400)
    """
    return StringEvaluator(
        correct_answer=correct_answer,
        case_sensitive=case_sensitive,
        trim_whitespace=trim_whitespace,
        regex_match=regex_match,
        **options,
    )


def fun_cmp(
    correct_answer: str | Formula,
    var: str | list[str] = "x",
    limits: list[tuple[float, float]] | None = None,
    num_points: int = 5,
    tolerance: float = 0.001,
    tolerance_mode: str = "relative",
    **options: Any,
) -> FormulaEvaluator:
    """
    Create formula answer checker.

    Args:
        correct_answer: Correct formula (string or Formula object)
        var: Variable name(s) (default "x")
        limits: Test range for each variable [(min, max), ...]
        num_points: Number of random test points (default 5)
        tolerance: Numerical tolerance (default 0.001)
        tolerance_mode: "relative", "absolute", or "sigfigs"
        **options: Additional evaluator options

    Returns:
        FormulaEvaluator configured with specified options

    Examples:
        >>> ANS(fun_cmp("x^2 + 1", var="x"))
        >>> ANS(fun_cmp("sin(x)*cos(y)", var=["x", "y"]))
        >>> ANS(fun_cmp("x^2", var="x", limits=[(-5, 5)]))

    Reference:
        PGanswermacros.pl::fun_cmp (lines 1800-2200)
    """
    # Normalize var to list
    variables = var if isinstance(var, list) else [var]

    # Don't parse - let FormulaEvaluator handle it
    return FormulaEvaluator(
        correct_answer=correct_answer,  # Pass string directly
        variables=variables,
        limits=limits,
        num_test_points=num_points,
        tolerance=tolerance,
        tolerance_mode=tolerance_mode,
        **options,
    )


def radio_cmp(
    correct_answer: str | int,
    **options: Any,
) -> StringEvaluator:
    """
    Create radio button answer checker.

    Args:
        correct_answer: Correct choice (letter like "A" or index like 0)
        **options: Additional options

    Returns:
        StringEvaluator for radio button choice

    Examples:
        >>> ANS(radio_cmp("A"))
        >>> ANS(radio_cmp(0))  # Index-based

    Reference:
        PGanswermacros.pl::radio_cmp
    """
    # Convert to string if integer index
    if isinstance(correct_answer, int):
        correct_answer = str(correct_answer)

    return StringEvaluator(
        correct_answer=correct_answer,
        case_sensitive=False,
        trim_whitespace=True,
        **options,
    )


def checkbox_cmp(
    correct_answers: list[str] | list[int],
    **options: Any,
) -> StringEvaluator:
    """
    Create checkbox answer checker (multiple correct answers).

    Args:
        correct_answers: List of correct choices
        **options: Additional options

    Returns:
        StringEvaluator for checkbox choices

    Examples:
        >>> ANS(checkbox_cmp(["A", "C"]))
        >>> ANS(checkbox_cmp([0, 2]))  # Index-based

    Reference:
        PGanswermacros.pl::checkbox_cmp
    """
    # For now, simple list comparison
    # Full implementation would use MultiAnswer coordination
    correct_str = ",".join(str(ans) for ans in sorted(correct_answers))

    return StringEvaluator(
        correct_answer=correct_str,
        case_sensitive=False,
        trim_whitespace=True,
        **options,
    )


# Aliases for backwards compatibility
def number_cmp(*args: Any, **kwargs: Any) -> NumericEvaluator:
    """Alias for num_cmp()."""
    return num_cmp(*args, **kwargs)


def function_cmp(*args: Any, **kwargs: Any) -> FormulaEvaluator:
    """Alias for fun_cmp()."""
    return fun_cmp(*args, **kwargs)


def formula_cmp(*args: Any, **kwargs: Any) -> FormulaEvaluator:
    """Alias for fun_cmp()."""
    return fun_cmp(*args, **kwargs)
