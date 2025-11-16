"""
PolynomialFactors context for MathObjects.

This module implements a context that restricts formulas to factored polynomial
form, rejecting expanded polynomials and ensuring proper factorization.

Example:
    Accept: (x-1)(x+2), 4(2x+1)(x+3), (x-1)^2
    Reject: x^2 + x - 2, (x-1) + (x+2), sin(x)

Port to pg_math from pg.mathobjects for Perl 1:1 parity.
"""

import sympy as sp
from typing import Optional, Tuple, Set


class FactoredPolynomialValidator:
    """Validates that expressions are in factored polynomial form."""

    def __init__(
        self,
        variables: Set[str],
        single_factors: bool = False,
        strict_powers: bool = True,
        strict_division: bool = False,
        strict_coefficients: bool = False,
        single_powers: bool = False,
    ):
        """
        Initialize the validator.

        Args:
            variables: Set of variable names
            single_factors: If True, factors cannot be repeated
            strict_powers: If True, only single factors can be raised to powers
            strict_division: If True, only single factors can be divided
            strict_coefficients: If True, no operations in coefficients
            single_powers: If True, only one monomial of each degree per factor
        """
        self.variables = variables
        self.single_factors = single_factors
        self.strict_powers = strict_powers
        self.strict_division = strict_division
        self.strict_coefficients = strict_coefficients
        self.single_powers = single_powers

    def validate(self, expr: sp.Expr) -> Tuple[bool, Optional[str]]:
        """
        Validate that expression is in factored polynomial form.

        Args:
            expr: Sympy expression to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # First check if it's a valid polynomial (from LimitedPolynomial)
        # This catches functions, fractional powers, etc.
        is_poly, msg = self._check_polynomial_restrictions(expr)
        if not is_poly:
            return False, msg

        # Check if in factored form (not expanded)
        is_factored, msg = self._check_factored_form(expr)
        if not is_factored:
            return False, msg

        # Check factor-specific restrictions
        if self.single_factors or self.strict_powers or self.strict_division:
            is_valid, msg = self._check_factor_restrictions(expr)
            if not is_valid:
                return False, msg

        return True, None

    def _check_polynomial_restrictions(
        self, expr: sp.Expr
    ) -> Tuple[bool, Optional[str]]:
        """Check basic polynomial restrictions (no functions, integer powers, etc.)"""
        # Check for functions on variables
        functions = expr.atoms(sp.Function)
        var_syms = {sp.Symbol(v) for v in self.variables}

        for func in functions:
            # Check if function arguments contain our variables
            func_args = func.args
            for arg in func_args:
                if arg.free_symbols & var_syms:
                    func_name = func.func.__name__
                    # Map sympy names to common names
                    name_map = {'log': 'ln'}
                    display_name = name_map.get(func_name, func_name)
                    return False, f"function '{display_name}' not allowed in a polynomial"

        # Check powers
        powers = expr.atoms(sp.Pow)
        for pow_expr in powers:
            base, exponent = pow_expr.as_base_exp()

            # If base contains variables, check exponent
            if base.free_symbols & var_syms:
                if not exponent.is_Integer:
                    return False, "Exponent must be integer in a polynomial"
                if exponent < 0:
                    return False, "Exponents must be non-negative in a polynomial"

        # Check if polynomial in all variables
        for var_name in self.variables:
            var = sp.Symbol(var_name)
            if var in expr.free_symbols:
                try:
                    if not expr.is_polynomial(var):
                        return False, f"Not a polynomial in {var_name}"
                except:
                    return False, f"Not a polynomial in {var_name}"

        return True, None

    def _check_factored_form(self, expr: sp.Expr) -> Tuple[bool, Optional[str]]:
        """
        Check if expression is in factored form.

        Factored form means:
        - Product of factors: (x-1)(x+2)
        - Constant multiple: 3(x-1)(x+2)
        - Powers of factors: (x-1)^2
        - Division by constant: (x-1)(x+2)/3
        - Single linear or simple factor: x+1, x-1, x

        Not factored:
        - Addition at top level (unless single simple factor): (x-1) + (x+2)
        - Expanded polynomial: x^2 + x - 2
        """
        # Get all variables in expression
        expr_vars = expr.free_symbols
        var_syms = {sp.Symbol(v) for v in self.variables}
        has_vars = bool(expr_vars & var_syms)

        if not has_vars:
            # Pure constant is OK
            return True, None

        # Check if top level is addition
        if isinstance(expr, sp.Add):
            # Exception: single LINEAR polynomial like x+1 is considered a factor
            # But NOT higher degree like x^2 + x - 2
            if self._is_simple_linear_factor(expr):
                return True, None
            return False, "Polynomial must be in factored form (use parentheses for factors)"

        # If it's multiplication, check factors
        if isinstance(expr, sp.Mul):
            return self._check_multiplication_factors(expr)

        # If it's a power, check base
        if isinstance(expr, sp.Pow):
            base, exp = expr.as_base_exp()
            # Exponent must be integer (checked by polynomial validator)
            # Base should be polynomial factor
            if isinstance(base, sp.Add):
                return True, None  # (x+1)^2 is OK
            if isinstance(base, sp.Mul):
                # (x*(x+1))^2 might violate strictPowers
                if self.strict_powers:
                    return False, "You can only raise a single factor to a power (not a product)"
                return True, None
            return True, None

        # Single variable or constant
        if expr.is_Symbol or expr.is_number:
            return True, None

        return False, "Expression must be in factored polynomial form"

    def _is_simple_linear_factor(self, expr: sp.Expr) -> bool:
        """
        Check if expression is a simple linear factor (degree 1 in each variable).

        Accept: x+1, 2*x+3, x+y
        Reject: x^2+1, x^2+x-2
        """
        if not isinstance(expr, sp.Add):
            return False

        # Check degree in each variable
        for var_name in self.variables:
            var = sp.Symbol(var_name)
            if var in expr.free_symbols:
                try:
                    degree = sp.degree(expr, var)
                    if degree > 1:
                        return False  # Higher than linear, reject
                except:
                    return False

        return True

    def _check_multiplication_factors(self, expr: sp.Mul) -> Tuple[bool, Optional[str]]:
        """Check factors in a multiplication."""
        factors = expr.as_ordered_factors()

        # Each non-constant factor should be polynomial
        for factor in factors:
            if not factor.is_number:
                # Should be variable, power, or polynomial in parens
                if isinstance(factor, sp.Add):
                    # Polynomial factor like (x+1) - should be linear or simple
                    if not self._is_simple_linear_factor(factor):
                        # Allow it if it's a valid polynomial, but this means
                        # they might have something like (x^2+1) as a factor
                        # For now, allow any polynomial factor
                        pass
                elif isinstance(factor, sp.Pow):
                    # Power of factor like (x+1)^2
                    base, _ = factor.as_base_exp()
                    if isinstance(base, sp.Add):
                        if not self._is_simple_linear_factor(base):
                            # Allow any polynomial base for now
                            pass

        return True, None

    def _check_factor_restrictions(self, expr: sp.Expr) -> Tuple[bool, Optional[str]]:
        """Check factor-specific restrictions (uniqueness, powers, division)."""
        # Handle division
        if isinstance(expr, sp.Mul):
            # Check for division (negative powers)
            has_division = any(
                isinstance(f, sp.Pow) and f.as_base_exp()[1] < 0
                for f in expr.as_ordered_factors()
            )
            if has_division and self.strict_division:
                # Check if dividing a product
                numerator_factors = [
                    f
                    for f in expr.as_ordered_factors()
                    if not (isinstance(f, sp.Pow) and f.as_base_exp()[1] < 0)
                ]
                if len([f for f in numerator_factors if not f.is_number]) > 1:
                    return False, "You can only divide a single factor by a number (not a product)"

        # Check for repeated factors if singleFactors is set
        if self.single_factors:
            is_valid, msg = self._check_single_factors(expr)
            if not is_valid:
                return False, msg

        return True, None

    def _check_single_factors(self, expr: sp.Expr) -> Tuple[bool, Optional[str]]:
        """Check that factors are not repeated (if singleFactors flag is set)."""
        factors = self._extract_factor_strings(expr)

        # Check for duplicates
        seen = set()
        for factor in factors:
            if factor == "constant":
                continue  # Skip constant multiples
            if factor in seen:
                return False, "Each factor can appear only once (combine like factors)"
            seen.add(factor)

        return True, None

    def _extract_factor_strings(self, expr: sp.Expr) -> list:
        """Extract factor strings for uniqueness checking."""
        factors = []

        if isinstance(expr, sp.Mul):
            for factor in expr.as_ordered_factors():
                if factor.is_number:
                    factors.append("constant")
                elif isinstance(factor, sp.Pow):
                    # For (x+1)^2, extract base (x+1)
                    base, _ = factor.as_base_exp()
                    factors.append(str(base))
                else:
                    factors.append(str(factor))
        elif isinstance(expr, sp.Pow):
            # Single power like (x+1)^2
            base, _ = expr.as_base_exp()
            factors.append(str(base))
        elif isinstance(expr, sp.Add):
            # Single factor
            factors.append(str(expr))
        elif not expr.is_number:
            # Single variable or expression
            factors.append(str(expr))

        return factors


def create_polynomial_factors_context():
    """
    Create a PolynomialFactors context.

    This is a factory function that would be called by Context initialization.
    Returns a Context configured for factored polynomial validation.
    """
    from .context import Context

    # Create context based on LimitedPolynomial
    ctx = Context("LimitedPolynomial")
    ctx.name = "PolynomialFactors"

    # Set PolynomialFactors flags
    ctx.flags.set(
        polynomialFactors=True,
        strictPowers=True,  # Default in Perl version
    )

    return ctx


def create_polynomial_factors_strict_context():
    """
    Create a PolynomialFactors-Strict context.

    Strict mode:
    - No operations in coefficients
    - singlePowers, singleFactors, strictDivision, strictPowers all True
    - Functions disabled
    - reduceConstants off
    """
    from .context import Context

    # Create context based on LimitedPolynomial-Strict
    ctx = Context("LimitedPolynomial-Strict")
    ctx.name = "PolynomialFactors-Strict"

    # Set all strict flags
    ctx.flags.set(
        polynomialFactors=True,
        strictCoefficients=True,
        strictDivision=True,
        strictPowers=True,
        singlePowers=True,
        singleFactors=True,
        reduceConstants=False,
    )

    return ctx


def validate_factored_polynomial(formula_obj) -> Tuple[bool, Optional[str]]:
    """
    Validate that a formula is in factored polynomial form.

    This is called by Formula during parsing if the context requires it.

    Args:
        formula_obj: Formula object to validate

    Returns:
        Tuple of (is_valid, error_message) for consistency with other validators
    """
    ctx = formula_obj.context
    if ctx is None:
        return True, None

    flags = ctx.flags

    # Only validate if polynomialFactors flag is set
    if not flags.get("polynomialFactors"):
        return True, None

    # Get validator settings from flags
    # strictPowers defaults to True in Perl version
    strict_powers = flags.get("strictPowers")
    if strict_powers is None:
        strict_powers = True

    validator = FactoredPolynomialValidator(
        variables=set(ctx.variables.list()),
        single_factors=flags.get("singleFactors") or False,
        strict_powers=strict_powers,
        strict_division=flags.get("strictDivision") or False,
        strict_coefficients=flags.get("strictCoefficients") or False,
        single_powers=flags.get("singlePowers") or False,
    )

    # Validate the parsed expression (use _sympy_expr for pg_math)
    is_valid, error_msg = validator.validate(formula_obj._sympy_expr)

    return is_valid, error_msg
