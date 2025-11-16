"""
Context-related macros for WeBWorK.

This package provides implementations of various context-related macros:
- contextUnits.pl - Units context with support for physical units
- (Future: contextFraction.pl, contextLimitedPolynomial.pl, etc.)
"""

from pg.math.context import get_context as Context
from .context_units import Context_Units, UnitsContext, UNIT_DEFINITIONS

__all__ = [
    'Context',
    'Context_Units',
    'UnitsContext',
    'UNIT_DEFINITIONS',
]
