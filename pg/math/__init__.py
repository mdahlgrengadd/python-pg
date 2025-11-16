"""
pg_math - MathObjects system for WeBWorK PG

Intelligent mathematical value types with:
- Type promotion
- Operator overloading
- Fuzzy comparison
- Multiple output formats

Reference: lib/Value.pm and lib/Value/*.pm in legacy Perl codebase
"""

from .collections import List, String
from .compute import Compute
from .context import Context, get_context, get_current_context
from .formula import Formula
from .formula_up_to_constant import FormulaUpToConstant
from .fraction import Fraction
from .geometric import Matrix, Point, Vector, norm
from .limited_polynomial import create_limited_polynomial_context, validate_polynomial_formula
from .numeric import Complex, Infinity, Real
from .polynomial_factors import (
    create_polynomial_factors_context,
    create_polynomial_factors_strict_context,
    validate_factored_polynomial,
)
from .sets import Interval, Set, Union
from .value import MathValue, ToleranceMode, TypePrecedence

__all__ = [
    "MathValue",
    "TypePrecedence",
    "ToleranceMode",
    "Real",
    "Complex",
    "Infinity",
    "Fraction",
    "Point",
    "Vector",
    "Matrix",
    "norm",
    "List",
    "String",
    "Interval",
    "Set",
    "Union",
    "Formula",
    "FormulaUpToConstant",
    "Context",
    "get_context",
    "get_current_context",
    "Compute",
    "create_limited_polynomial_context",
    "validate_polynomial_formula",
    "create_polynomial_factors_context",
    "create_polynomial_factors_strict_context",
    "validate_factored_polynomial",
]
