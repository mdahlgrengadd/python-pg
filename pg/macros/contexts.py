"""
Context-related macro stubs for PG problems.

Provides stub implementations for various context macros like:
- contextFraction.pl
- contextIntegerFunctions.pl
- contextLimitedPolynomial.pl
etc.
"""


def Context(name: str = 'Numeric'):
    """
    Load or switch to a context.

    Common contexts:
    - 'Numeric' (default)
    - 'Fraction', 'Fraction-NoDecimals'
    - 'LimitedFraction', 'LimitedProperFraction'
    - 'Complex'
    - 'Point', 'Vector', 'Matrix'
    - 'Interval'
    - 'String'

    Delegates to pg_math.context.get_context for proper context switching.
    """
    # Import get_context to properly set the current context
    from pg.math.context import get_context

    # Use get_context which properly sets and returns the context
    return get_context(name)


# Export for macro loading
__all__ = ['Context']
