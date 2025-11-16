"""
PGstandard.pl - Core PG functions

Reference: macros/core/PGstandard.pl (1,234 lines)
"""

from typing import Any, List, Tuple, Union
import math


def TEXT(*args: Any) -> str:
    """
    Accumulate text for problem statement.
    
    Reference: PGstandard.pl::TEXT
    """
    # Convert all arguments to strings and concatenate
    return "".join(str(arg) for arg in args)


def BEGIN_TEXT() -> str:
    """
    Marker for beginning of text block.
    This is typically handled by preprocessor.
    
    Reference: PGstandard.pl::BEGIN_TEXT
    """
    return ""


def END_TEXT() -> str:
    """
    Marker for end of text block.
    
    Reference: PGstandard.pl::END_TEXT
    """
    return ""


def ANS(*evaluators: Any) -> None:
    """
    Register answer evaluators.
    
    Reference: PGstandard.pl::ANS
    """
    # In full implementation, this would register with problem environment
    # For now, placeholder
    pass


def NAMED_ANS(name: str, evaluator: Any) -> None:
    """
    Register named answer evaluator.
    
    Reference: PGstandard.pl::NAMED_ANS
    """
    pass


def image(filename: str, **options: Any) -> str:
    """
    Insert image.
    
    Reference: PGstandard.pl::image
    """
    width = options.get('width', '')
    height = options.get('height', '')
    alt = options.get('alt', '')
    
    attrs = []
    if width:
        attrs.append(f'width="{width}"')
    if height:
        attrs.append(f'height="{height}"')
    if alt:
        attrs.append(f'alt="{alt}"')
    
    attr_str = ' '.join(attrs)
    return f'<img src="{filename}" {attr_str}>'


def bold(text: str) -> str:
    """Return bolded text."""
    return f"**{text}**"


def italic(text: str) -> str:
    """Return italicized text."""
    return f"*{text}*"


def underline(text: str) -> str:
    """Return underlined text."""
    return f"<u>{text}</u>"


def ans_rule(width: int = 20) -> str:
    """
    Create answer blank.
    
    Reference: PGstandard.pl::ans_rule
    """
    return f'<input type="text" size="{width}" class="pg-answer-blank">'


def solution(*args: Any) -> str:
    """
    Create solution section.
    
    Reference: PGstandard.pl::SOLUTION
    """
    content = "".join(str(arg) for arg in args)
    return f'<div class="solution">{content}</div>'


def hint(*args: Any) -> str:
    """
    Create hint section.
    
    Reference: PGstandard.pl::HINT
    """
    content = "".join(str(arg) for arg in args)
    return f'<div class="hint">{content}</div>'


# Random number functions

# Import random functions from pg_core (which uses the seeded RNG from PG environment)
from .pg_core import random, non_zero_random, list_random


def shuffle(*items):
    """
    Shuffle list items.
    
    Reference: PGstandard.pl::shuffle
    """
    import random as _random
    shuffled = list(items)
    _random.shuffle(shuffled)
    return shuffled


def random_subset(n: int, *items):
    """
    Select n random items from list without replacement.

    Can be called as:
    - random_subset(2, item1, item2, item3, ...) - variadic style
    - random_subset(2, [item1, item2, item3, ...]) - list style

    Reference: PGstandard.pl::random_subset
    """
    import random as _random

    # Handle case where a single list is passed (from Perl array conversion)
    if len(items) == 1 and isinstance(items[0], (list, tuple)):
        items = items[0]

    if n < 0 or n > len(items):
        # Match Perl behavior: if sample size is invalid, return what we can
        # or raise an error if it's clearly wrong
        if n < 0:
            raise ValueError("Sample size cannot be negative")
        # Sample all items if n is larger than population (graceful degradation)
        return _random.sample(items, len(items))

    return _random.sample(items, n)


# String formatting and polynomial utilities


def nicestring(
    coefficients: List[Union[int, float]],
    variables: List[str] | None = None,
) -> str:
    """
    Format polynomial coefficients as a nice string expression.

    Args:
        coefficients: List of polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]
        variables: Optional list of variable names. If None, generates x^(n-1), x^(n-2), ..., x, 1

    Returns:
        Formatted string like "x^2 + 2x + 1" or "2x^2 - 3x + 5"

    Reference: PGbasicmacros.pl::nicestring

    Example:
        nicestring([1, 2, 1]) → "x^2 + 2x + 1"
        nicestring([1, 0, -1]) → "x^2 - 1"
        nicestring([2, -3, 1]) → "2x^2 - 3x + 1"
    """
    coefs = list(coefficients)
    n = len(coefs)

    # Generate default variables if not provided
    if variables is None:
        variables = []
        for j in range(1, n - 1):
            variables.append(f"x^{n - j}")
        if n >= 2:
            variables.append("x")
        variables.append("")

    # Find first non-zero coefficient
    k = 0
    while k < n and coefs[k] == 0:
        k += 1

    # All zeros
    if k == n:
        return "0"

    # Build result string starting with first non-zero term
    result = ""
    if coefs[k] == 1:
        result = variables[k] if variables[k] else "1"
    elif coefs[k] == -1:
        result = f"-{variables[k]}" if variables[k] else "-1"
    else:
        result = f"{coefs[k]} {variables[k]}".strip()

    # Add remaining terms
    for j in range(k + 1, n):
        if coefs[j] != 0:
            if coefs[j] == 1:
                term = f"+ {variables[j]}" if variables[j] else "+ 1"
            elif coefs[j] == -1:
                term = f"- {variables[j]}" if variables[j] else "- 1"
            elif coefs[j] > 0:
                term = f"+ {coefs[j]} {variables[j]}".strip()
            else:
                term = f"- {abs(coefs[j])} {variables[j]}".strip()

            result += f" {term}" if result else term

    return result


def TeX(latex_str: str) -> str:
    """
    Escape special characters in LaTeX string for safe inclusion in output.

    Converts TeX code to safe display format.

    Reference: PGbasicmacros.pl (TeX handling)

    Example:
        TeX("x_1") → "x_1"  (no escaping needed for most math)
    """
    # In context of PGML/modern PG, most TeX is safe
    # This is mainly for legacy code - return as-is in most cases
    # Only escape if needed for display mode
    return latex_str


def EV2(string: str) -> str:
    """
    Evaluate embedded code in \\{ \\} and \\< \\> delimiters.

    Does NOT evaluate variable substitution or outer context.

    Reference: PGbasicmacros.pl::EV2

    Note:
        In Python version, this is mainly for legacy code.
        Modern PGML handles most of this automatically.
    """
    # This is a simplified version - full EV2 requires safe eval context
    # For now, return as-is since modern PG doesn't need this
    return string


def EV3(string: str) -> str:
    """
    Evaluate variables and embedded code in problem string.

    Processes \\{ \\} code blocks with variable interpolation.

    Reference: PGbasicmacros.pl::EV3

    Note:
        In Python version, this is mainly for legacy code.
        Modern PGML handles most of this automatically.
    """
    # This requires safe evaluation context which is complex in Python
    # For now, return as-is
    return string


def display_matrix(matrix: List[List[Union[int, float]]], **options) -> str:
    """
    Format matrix for display.

    Args:
        matrix: 2D list of numbers
        **options: display options (not fully implemented)

    Returns:
        LaTeX matrix or HTML table representation

    Reference: PGbasicmacros.pl (via context)

    Example:
        display_matrix([[1, 2], [3, 4]])
    """
    if not matrix:
        return ""

    # Simple LaTeX matrix representation
    rows = []
    for row in matrix:
        row_str = " & ".join(str(x) for x in row)
        rows.append(row_str)

    # Return LaTeX array format
    return "\\begin{pmatrix}\n" + " \\\\\n".join(rows) + "\n\\end{pmatrix}"


def display_matrix_mm(matrix: List[List[Union[int, float]]], **options) -> str:
    """
    Format matrix for display (mm = math mode).

    Alternative format for matrix display.

    Reference: PGbasicmacros.pl
    """
    return display_matrix(matrix, **options)


def display_matrix_mr(matrix: List[List[Union[int, float]]], **options) -> str:
    """
    Format matrix for display (mr = matrix row format).

    Reference: PGbasicmacros.pl
    """
    return display_matrix(matrix, **options)
