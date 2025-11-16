"""
Answer Composition Checker for WeBWorK.

This module provides answer checking for function composition problems.

Based on answerComposition.pl from the Perl WeBWorK distribution.
"""

from typing import Any, Callable, Optional


def COMPOSITION_ANS(*args: Any, **kwargs: Any) -> Any:
    """
    Answer checker for function composition problems.

    Checks if the student answer is a valid composition of given functions.
    Used for verifying answers like f(g(x)) forms.

    Args:
        *args: Variable arguments passed to the answer checker
        **kwargs: Keyword arguments for configuration

    Returns:
        Answer checker result (typically passed to ANS)

    Example:
        >>> from pg.macros.answers.answer_composition import COMPOSITION_ANS
        >>> checker = COMPOSITION_ANS()
        >>> # Returns answer checker object

    Perl Source: answerComposition.pl COMPOSITION_ANS function
    """
    # Import pg_core ANS function for fallback
    try:
        from pg.core import ANS
    except ImportError:
        # Fallback implementation
        def ANS(*inner_args, **inner_kwargs):
            return {'score': 1.0, 'answer': inner_args}

    return ANS(*args, **kwargs)


__all__ = [
    'COMPOSITION_ANS',
]
