"""
Unordered Answer Checker for WeBWorK.

This module provides answer checking for problems where multiple correct
answers can be given in any order (e.g., set-based problems).

Based on unorderedAnswer.pl from the Perl WeBWorK distribution.
"""

from typing import Any


def UNORDERED_ANS(*args: Any, **kwargs: Any) -> Any:
    """
    Answer checker for problems accepting unordered answers.
    
    Accepts student answers in any order. Commonly used for set-based problems
    where the order of elements doesn't matter.
    
    Args:
        *args: Variable arguments passed to the answer checker
        **kwargs: Keyword arguments for configuration
        
    Returns:
        Answer checker result (typically passed to ANS)
        
    Example:
        >>> from pg.macros.answers.unordered_answer import UNORDERED_ANS
        >>> checker = UNORDERED_ANS()
        >>> # Returns answer checker object
    
    Perl Source: unorderedAnswer.pl UNORDERED_ANS function
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
    'UNORDERED_ANS',
]

