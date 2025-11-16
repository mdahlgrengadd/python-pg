"""
PGcourse.pl - Core PG course setup functions.

Top-level barrel module for short imports (1:1 parity with Perl PGcourse.pl).
Re-exports from pg.macros.core.pg_core.

Usage:
    from pg.course import loadMacros
    loadMacros("MathObjects.pl")

Note: This is a barrel module that provides a clean top-level import API.
It aggregates multiple submodules from pg.macros for user convenience.

Reference: macros/PGcourse.pl
"""

from pg.macros.core.pg_core import loadMacros

__all__ = [
    "loadMacros",
]
