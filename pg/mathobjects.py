"""
MathObjects.pl - MathObjects for problem authoring.

Top-level barrel module for short imports (1:1 parity with Perl MathObjects.pl).
Re-exports from pg.macros.MathObjects which provides complete MathObjects from pg.math.

Usage:
    from pg.mathobjects import Context, Formula, Real, Compute
    Context("Numeric")
    f = Formula("x^2")

Note: This is a barrel module that provides a clean top-level import API.
It aggregates multiple submodules from pg.macros for user convenience.

Reference: macros/core/MathObjects.pl
"""

from pg.macros.MathObjects import (
    Context,
    Formula,
    Real,
    Complex,
    Compute,
    Vector,
    Point,
    Interval,
    Set,
    String,
    List,
    Matrix,
    Fraction,
)

# Also export Infinity and Union from pg.math
from pg.math import Infinity, Union

# Define imaginary units for convenience
# These are based on the current context's constants
# In Numeric context, i is the imaginary unit (Complex(0, 1))
# In Vector context, i, j, k are unit vectors
try:
    # Try to get i from the current context
    _ctx = Context()
    if 'i' in _ctx.constants:
        i = _ctx.constants.get('i')
    else:
        # Fallback: define as Complex imaginary unit
        i = Complex(0, 1)
except:
    # Fallback: define as Complex imaginary unit
    i = Complex(0, 1)

__all__ = [
    "Context",
    "Formula",
    "Real",
    "Complex",
    "Compute",
    "Vector",
    "Point",
    "Interval",
    "Set",
    "String",
    "List",
    "Matrix",
    "Fraction",
    "i",
    "Union",
    "Infinity",
]
