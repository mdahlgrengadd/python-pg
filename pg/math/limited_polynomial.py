"""Limited Polynomial Context - Restricts formulas to polynomial form.

This module provides a context that only allows polynomial expressions,
rejecting functions, fractional powers, and other non-polynomial operations.

Reference: lib/pg/macros/parserLimitedPolynomial.pl in legacy Perl codebase

Usage:
    from pg.math import Context, Formula

    ctx = Context('LimitedPolynomial')
    f = Formula('x^2 + 2*x + 1', ['x'], ctx)  # OK
    g = Formula('sin(x)', ['x'], ctx)  # Error: functions not allowed
"""

import sympy as sp
from typing import Tuple

from .context import Context
from .formula import Formula


def create_limited_polynomial_context(strict: bool = False) -> Context:
    """Create a LimitedPolynomial context.

    Args:
        strict: If True, disallow operations in coefficients

    Returns:
        Context configured for polynomial expressions

    Reference: parserLimitedPolynomial.pl Context initialization
    """
    if strict:
        ctx = Context('LimitedPolynomial-Strict')
    else:
        ctx = Context('LimitedPolynomial')

    return ctx


class PolynomialValidator:
    """Validates that expressions are in polynomial form.

    Reference: parserLimitedPolynomial.pl validation logic
    """

    def __init__(self, strict: bool = False, single_powers: bool = False):
        """Initialize validator.

        Args:
            strict: If True, coefficients cannot contain operations
            single_powers: If True, powers must appear only once per variable
        """
        self.strict = strict
        self.single_powers = single_powers

    def validate(self, formula: Formula) -> Tuple[bool, str | None]:
        """Validate that a formula is a polynomial.

        Args:
            formula: Formula to validate

        Returns:
            (is_valid, error_message) tuple
        """
        if not hasattr(formula, '_sympy_expr') or formula._sympy_expr is None:
            return False, "Cannot validate formula without SymPy expression"

        variables = self._get_variables(formula)

        # Check for functions (sin, cos, etc.)
        is_valid, error = self._check_functions(formula._sympy_expr, variables)
        if not is_valid:
            return False, error

        # Check for fractional or negative powers
        is_valid, error = self._check_powers(formula._sympy_expr, variables)
        if not is_valid:
            return False, error

        # In strict mode, check coefficients
        if self.strict:
            is_valid, error = self._check_strict_coefficients(
                formula._sympy_expr, variables)
            if not is_valid:
                return False, error

        # Check single powers if required
        if self.single_powers:
            is_valid, error = self._check_single_powers(
                formula._sympy_expr, variables)
            if not is_valid:
                return False, error

        return True, None

    def _get_variables(self, formula: Formula) -> set:
        """Get the set of variables in the formula as SymPy Symbols."""
        if hasattr(formula, 'variables'):
            # Filter out constants and convert to SymPy Symbols
            context_vars = formula.context.variables.list() if formula.context else []
            var_names = set(v for v in formula.variables if v in context_vars)
            return {sp.Symbol(v) for v in var_names}
        return set()

    def _is_polynomial_tree(self, expr: sp.Expr, variables: set) -> Tuple[bool, str | None]:
        """Check if expression tree is polynomial."""
        # This is called by _check_functions and _check_powers
        return True, None

    def _check_functions(self, expr: sp.Expr, variables: set) -> Tuple[bool, str | None]:
        """Check that no non-polynomial functions are used.

        Reference: parserLimitedPolynomial.pl checkFunctions
        """
        # Get all functions in the expression
        for func in expr.atoms(sp.Function):
            func_name = func.func.__name__
            # Allow basic operations but not trig, log, exp, etc.
            disallowed = [
                'sin', 'cos', 'tan', 'sec', 'csc', 'cot',
                'asin', 'acos', 'atan', 'asec', 'acsc', 'acot',
                'sinh', 'cosh', 'tanh', 'sech', 'csch', 'coth',
                'asinh', 'acosh', 'atanh', 'asech', 'acsch', 'acoth',
                'ln', 'log', 'exp', 'sqrt', 'abs'
            ]
            if func_name.lower() in disallowed:
                # Use user-friendly name (log -> ln for error message)
                display_name = 'ln' if func_name.lower() == 'log' else func_name.lower()
                return False, f"function '{display_name}' is not allowed in polynomial expressions"

        # Also check for sqrt using Pow with 1/2
        for term in sp.preorder_traversal(expr):
            if isinstance(term, sp.Pow):
                base, exp = term.args
                # Check if exponent involves variables (not allowed in polynomials)
                if exp.free_symbols & variables:
                    return False, "Variables in exponents are not allowed in polynomials"
                # Check for fractional/negative powers of variables
                if base.free_symbols & variables:
                    try:
                        exp_val = float(exp)
                        if exp_val < 0:
                            return False, "exponent must be a non-negative integer in polynomial"
                        if not exp_val.is_integer():
                            return False, "exponent must be an integer"
                    except (TypeError, AttributeError):
                        # Symbolic exponent - only ok if not involving variables
                        if exp.free_symbols & variables:
                            return False, "Variables in exponents are not allowed"

        return True, None

    def _check_powers(self, expr: sp.Expr, variables: set) -> Tuple[bool, str | None]:
        """Check that all powers are non-negative integers.

        Reference: parserLimitedPolynomial.pl checkPowers
        """
        for term in sp.preorder_traversal(expr):
            if isinstance(term, sp.Pow):
                base, exp = term.args

                # Check if base contains variables
                if base.free_symbols & variables:
                    try:
                        exp_val = float(exp)
                        if exp_val < 0:
                            return False, "exponent must be a non-negative integer"
                        if abs(exp_val - round(exp_val)) > 1e-10:  # Not an integer
                            return False, "exponent must be an integer"
                    except (TypeError, ValueError, AttributeError):
                        # Can't evaluate to float - check if it's symbolic
                        if exp.free_symbols:
                            return False, "Variable exponents are not allowed in polynomials"

        return True, None

    def _check_strict_coefficients(self, expr: sp.Expr, variables: set) -> Tuple[bool, str | None]:
        """Check that coefficients don't contain operations (strict mode).

        Reference: parserLimitedPolynomial.pl checkCoefficients
        """
        # Expand the polynomial
        expanded = sp.expand(expr)

        # Get terms
        if isinstance(expanded, sp.Add):
            terms = expanded.args
        else:
            terms = [expanded]

        for term in terms:
            # Split into coefficient and variable part
            coeff, var_part = self._split_coefficient(term, variables)

            # Check if coefficient has operations
            if self._has_operations(coeff):
                return False, "Coefficients cannot contain operations in strict mode"

        return True, None

    def _split_coefficient(self, term: sp.Expr, variables: set) -> Tuple[sp.Expr, sp.Expr]:
        """Split term into coefficient and variable part.

        Reference: parserLimitedPolynomial.pl splitCoefficient
        """
        if isinstance(term, sp.Mul):
            coeff_parts = []
            var_parts = []
            for arg in term.args:
                if arg.free_symbols & variables:
                    var_parts.append(arg)
                else:
                    coeff_parts.append(arg)

            coeff = sp.Mul(*coeff_parts) if coeff_parts else sp.Integer(1)
            var_part = sp.Mul(*var_parts) if var_parts else sp.Integer(1)
            return coeff, var_part
        elif term.free_symbols & variables:
            return sp.Integer(1), term
        else:
            return term, sp.Integer(1)

    def _has_operations(self, expr: sp.Expr) -> bool:
        """Check if expression contains operations (add, mul, etc.).

        Reference: parserLimitedPolynomial.pl hasOperations
        """
        # Constants and numbers are ok
        if isinstance(expr, (sp.Integer, sp.Rational, sp.Float, sp.Number)):
            return False

        # Symbols (like pi, e) are ok
        if isinstance(expr, sp.Symbol):
            return False

        # Any operations (Add, Mul, Pow) mean it has operations
        if isinstance(expr, (sp.Add, sp.Mul, sp.Pow)):
            return True

        return False

    def _check_single_powers(self, expr: sp.Expr, variables: set) -> Tuple[bool, str | None]:
        """Check that each variable power appears at most once (singlePowers mode).

        Reference: parserLimitedPolynomial.pl checkSinglePowers
        """
        # Expand the polynomial
        expanded = sp.expand(expr)

        # Get terms
        if isinstance(expanded, sp.Add):
            terms = expanded.args
        else:
            terms = [expanded]

        # Track which powers we've seen for each variable
        seen_powers = {}

        for term in terms:
            # Get variable powers in this term
            for var in variables:
                # var is already a Symbol object
                power = sp.degree(term, var)

                if power > 0:
                    # Convert Symbol to string for dict key
                    var_name = str(var)
                    if var_name not in seen_powers:
                        seen_powers[var_name] = set()

                    if power in seen_powers[var_name]:
                        return False, f"Power {var_name}^{power} appears more than once (singlePowers mode)"

                    seen_powers[var_name].add(power)

        return True, None


def validate_polynomial_formula(formula: Formula) -> Tuple[bool, str | None]:
    """Validate formula is polynomial (used by Formula when parsing).

    Args:
        formula: Formula to validate

    Returns:
        (is_valid, error_message) tuple

    Reference: parserLimitedPolynomial.pl validate() function
    """
    # Check if context requires polynomial validation
    if not hasattr(formula, 'context') or formula.context is None:
        return True, None

    if not formula.context.flags.get('limitedPolynomial'):
        return True, None

    # Get validation settings from context
    strict = formula.context.flags.get('strictCoefficients') or False
    single_powers = formula.context.flags.get('singlePowers') or False

    # Validate
    validator = PolynomialValidator(strict, single_powers)
    return validator.validate(formula)
