"""
Fallback Utilities for WeBWorK.

This module provides miscellaneous fallback functions and utilities that are
not specific to other macro categories but are needed for general problem support.

These include utility functions like array operations, compatibility helpers,
and other general-purpose stubs.

Based on various PG core utilities.
"""

from typing import Any, List, Optional

# Import splice from the canonical implementation
from .array_utilities import splice


def new_match_list(*args: Any, **kwargs: Any) -> Any:
    """
    Create a matching list object for fill-in-the-blank matching problems.

    Args:
        *args: Configuration arguments
        **kwargs: Options

    Returns:
        A matching list object with default methods

    Perl Source: Matching list creation utility
    """
    class _MatchListStub:
        def __getattr__(self, name: str) -> Any:
            """Allow any method call - just return self for chaining."""
            def method(*a, **kw):
                return self
            return method

        def __iter__(self):
            return iter([])

    return _MatchListStub()


def pop_up_list_print_q(*args: Any, **kwargs: Any) -> str:
    """
    Print a pop-up list question.

    Args:
        *args: Question components
        **kwargs: Options

    Returns:
        HTML string for pop-up list

    Perl Source: Pop-up list printing utility
    """
    return ""


def undef() -> None:
    """
    Perl's undef - returns None.

    Used to indicate undefined/null values.

    Returns:
        None

    Perl Source: Perl undef compatibility
    """
    return None


__all__ = [
    'new_match_list',
    'pop_up_list_print_q',
    'splice',  # Re-exported from array_utilities
    'undef',
]
