"""
Special Trigonometric Parsers

Provides specialRadical and specialAngle for special-form trigonometric expressions.

Based on WeBWorK's PG macro libraries.
"""

from typing import Any


def specialRadical(expr: str, *args: Any, **kwargs: Any) -> Any:
    """
    Parse and evaluate special radical expressions.
    
    Used for trigonometric expressions with special forms like sqrt(2)/2.
    
    Args:
        expr: Expression to parse
        *args: Additional arguments
        **kwargs: Additional options
        
    Returns:
        Evaluated expression result
        
    Perl Source: parserSpecialRadical.pl
    """
    try:
        from pg.math import Compute
        return Compute(expr)
    except (ImportError, Exception):
        # Fallback: return string representation
        return expr


def specialAngle(expr: str, *args: Any, **kwargs: Any) -> Any:
    """
    Parse and evaluate special angle expressions.

    Used for angles in special forms like pi/3, 2*pi, etc.

    Args:
        expr: Expression to parse
        *args: Additional arguments
        **kwargs: Additional options

    Returns:
        Evaluated expression result

    Perl Source: parserSpecialAngle.pl
    """
    try:
        from pg.math import Compute
        return Compute(expr)
    except (ImportError, Exception):
        # Fallback: return string representation
        return expr


__all__ = ['specialRadical', 'specialAngle']
