"""
PGML.pl - PG Markup Language.

This module provides 1:1 parity with the Perl PGML.pl macro file.
Re-exports the PGML function from pg.macros.core.pgml.

Reference: macros/core/PGML.pl
"""

from .core.pgml import PGML

__all__ = ["PGML"]

