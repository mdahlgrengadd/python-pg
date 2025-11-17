"""
Fraction type for MathObjects.

Implements a Fraction that stores numerator and denominator as integers,
with support for reduction, mixed numbers, and various strictness options.

Reference: macros/contexts/contextFraction.pl, lib/Fraction.pm
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from .numeric import Real
from .value import MathValue, ToleranceMode, TypePrecedence


def gcd(a: int, b: int) -> int:
    """
    Greatest Common Divisor.

    Reference: contextFraction.pl::gcd
    """
    a, b = abs(a), abs(b)
    if a < b:
        a, b = b, a
    if b == 0:
        return a
    r = a % b
    while r != 0:
        a, b = b, r
        r = a % b
    return b


def lcm(a: int, b: int) -> int:
    """
    Least Common Multiple.

    Reference: contextFraction.pl::lcm
    """
    return (a // gcd(a, b)) * b


def reduce_fraction(num: int, den: int) -> tuple[int, int]:
    """
    Reduce fraction to lowest terms.

    Ensures denominator is positive.

    Reference: contextFraction.pl::reduce
    """
    if den < 0:
        num, den = -num, -den
    g = gcd(num, den)
    return (num // g, den // g)


class Fraction(MathValue):
    """
    Fraction represents a rational number as numerator/denominator.

    Examples:
        >>> Fraction(1, 2)  # 1/2
        >>> Fraction(3, 4)  # 3/4
        >>> Fraction(5, 2)  # 5/2 or 2 1/2

    Reference: macros/contexts/contextFraction.pl
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    numerator: int = Field(description="The numerator")
    denominator: int = Field(description="The denominator")
    context: Any = Field(default=None, description="The evaluation context")
    type_precedence: TypePrecedence = Field(default=TypePrecedence.FRACTION, init=False)

    def __init__(
        self,
        num: int | float,
        den: int = 1,
        context: Any = None,
        reduce: bool = True,
        **kwargs
    ):
        """
        Create a Fraction.

        Args:
            num: Numerator (or single value if den=1)
            den: Denominator (default 1)
            context: Mathematical context
            reduce: Whether to reduce to lowest terms (default True)

        Reference: lib/Fraction.pm::new
        """
        # If given a float, convert to fraction
        if isinstance(num, float) and den == 1:
            # Multiply by powers of 10 until we get an integer
            temp_den = 1
            temp_num = num
            while abs(temp_num - round(temp_num)) > 1e-9 and temp_den < 1e10:
                temp_num *= 10
                temp_den *= 10
            num = int(round(temp_num))
            den = temp_den

        num = int(num)
        den = int(den)

        if den == 0:
            raise ValueError("Fraction denominator cannot be zero")

        # Reduce if requested
        if reduce:
            num, den = reduce_fraction(num, den)

        super().__init__(numerator=num, denominator=den, context=context, **kwargs)

    @property
    def num(self) -> int:
        """Get numerator (backward compatibility property)."""
        return self.numerator

    @property
    def den(self) -> int:
        """Get denominator (backward compatibility property)."""
        return self.denominator

    @property
    def _num(self) -> int:
        """Get numerator (internal accessor for backward compatibility)."""
        return self.numerator

    @property
    def _den(self) -> int:
        """Get denominator (internal accessor for backward compatibility)."""
        return self.denominator

    def reduce(self) -> Fraction:
        """
        Reduce fraction to lowest terms.

        Returns new reduced Fraction.

        Reference: lib/Fraction.pm::reduce
        """
        num, den = reduce_fraction(self._num, self._den)
        return Fraction(num, den, self.context, reduce=False)

    def is_reduced(self) -> bool:
        """
        Check if fraction is in lowest terms.

        Reference: contextFraction.pl
        """
        return gcd(abs(self._num), abs(self._den)) == 1

    def promote(self, other: MathValue) -> MathValue:
        """Fractions promote to higher precedence."""
        if isinstance(other, Fraction):
            return self
        # Convert Real to Fraction
        if isinstance(other, Real):
            return self
        return self

    def compare(
        self, other: MathValue, tolerance: float = 0.001, mode: str = ToleranceMode.RELATIVE
    ) -> bool:
        """
        Compare two fractions for equality.

        Reference: contextFraction.pl
        """
        if isinstance(other, Fraction):
            # Reduce both and compare
            self_reduced = self.reduce()
            other_reduced = other.reduce()
            return (
                self_reduced._num == other_reduced._num
                and self_reduced._den == other_reduced._den
            )
        elif isinstance(other, Real):
            # Compare as decimals with fraction tolerance
            frac_tolerance = 1e-10
            if self.context:
                frac_tolerance = self.context.flags.get(
                    'fractionTolerance', 1e-10)
            self_value = self._num / self._den
            other_value = other.value
            return abs(self_value - other_value) <= frac_tolerance
        return False

    def to_real(self) -> Real:
        """Convert to Real number."""
        return Real(self._num / self._den)

    def to_string(self) -> str:
        """
        Convert to string representation.

        Handles mixed numbers if showMixedNumbers flag is set.

        Reference: contextFraction.pl
        """
        if self.context and self.context.flags.get('showMixedNumbers'):
            return self._to_mixed_string()
        return f"{self._num}/{self._den}"

    def _to_mixed_string(self) -> str:
        """Convert to mixed number string (e.g., "2 1/2")."""
        if abs(self._num) < abs(self._den):
            # Proper fraction
            return f"{self._num}/{self._den}"

        # Improper fraction - convert to mixed
        whole = self._num // self._den
        remainder = abs(self._num % self._den)

        if remainder == 0:
            return str(whole)
        return f"{whole} {remainder}/{abs(self._den)}"

    def to_tex(self) -> str:
        """Convert to LaTeX representation."""
        if self.context and self.context.flags.get('showMixedNumbers'):
            # Mixed number form
            if abs(self._num) < abs(self._den):
                return f"\\frac{{{self._num}}}{{{self._den}}}"

            whole = self._num // self._den
            remainder = abs(self._num % self._den)

            if remainder == 0:
                return str(whole)
            return f"{whole}\\,\\frac{{{remainder}}}{{{abs(self._den)}}}"

        return f"\\frac{{{self._num}}}{{{self._den}}}"

    def to_python(self) -> float:
        """Convert to Python float."""
        return self._num / self._den

    def __str__(self) -> str:
        """String representation for display - returns fraction notation."""
        return self.to_string()

    def __repr__(self) -> str:
        """Developer representation showing constructor."""
        return f"Fraction({self._num}, {self._den})"

    def cmp(self, **options):
        """
        Create an answer evaluator for this fraction.

        Options:
            studentsMustReduceFractions: Require reduced fractions (default from context)
            showFractionReductionWarnings: Show warnings for unreduced (default True)
            requireFraction: Must enter a fraction, not whole number (default False)
            requireProperFraction: Must enter a proper fraction (num < den) (default from context)
            strictFractions: Only integer division allowed (default from context)
            strictMinus: Strict minus handling (default False)
            strictMultiplication: Strict multiplication handling (default False)

        Reference: contextFraction.pl
        """
        # Create a simple answer checker that returns a checker function
        class FractionAnswerChecker:
            def __init__(self, correct_frac, opts):
                self.correct = correct_frac
                self.options = opts

            def check(self, student_answer):
                """Check student answer."""
                try:
                    import os
                    debug = not os.environ.get('PYPG_DISABLE_LOGGING')

                    # Parse student answer as Fraction
                    # IMPORTANT: Don't reduce during parsing - we need to check the form for reduction validation
                    from .compute import Compute
                    # We need to parse without reduction to check if the student reduced properly
                    # Parse as string and create Fraction with reduce=False
                    try:
                        # Try Compute first in case it's a formula
                        student_frac = Compute(
                            str(student_answer), self.correct.context)
                    except:
                        # If Compute fails, try direct parsing
                        student_frac = None

                    # If we got a Fraction from Compute, re-parse without reducing
                    if student_frac and isinstance(student_frac, Fraction):
                        # Parse the string directly to get unreduced form
                        answer_str = str(student_answer).strip()
                        if '/' in answer_str:
                            try:
                                parts = answer_str.split('/')
                                num_str, den_str = parts[0].strip(), parts[1].strip()
                                student_num = int(num_str)
                                student_den = int(den_str)
                                # Create fraction WITHOUT reducing
                                student_frac = Fraction(student_num, student_den, self.correct.context, reduce=False)
                            except:
                                # Fall back to the computed value
                                pass


                    if not isinstance(student_frac, Fraction):
                        return {'correct': False, 'score': 0.0, 'message': 'Answer must be a fraction'}

                    # Check requireFraction: reject whole numbers if set
                    if self.options.get('requireFraction', False):
                        if student_frac._den == 1:
                            return {'correct': False, 'score': 0.0, 'message': 'Your answer must be a fraction'}

                    # Check requireProperFraction: numerator must be < denominator
                    if self.options.get('requireProperFraction', False):
                        if abs(student_frac._num) >= abs(student_frac._den):
                            return {'correct': False, 'score': 0.0, 'message': 'Your answer must be a proper fraction'}

                    # Check if equal
                    if self.correct.compare(student_frac):
                        # Check reduction if required
                        if self.options.get('studentsMustReduceFractions', False):
                            is_reduced = student_frac.is_reduced()
                            if not is_reduced:
                                msg = 'Your answer is not reduced to lowest terms'
                                if self.options.get('showFractionReductionWarnings', True):
                                    return {'correct': False, 'score': 0.0, 'message': msg}
                                else:
                                    return {'correct': False, 'score': 0.0, 'message': ''}
                        return {'correct': True, 'score': 1.0, 'message': ''}
                    else:
                        return {'correct': False, 'score': 0.0, 'message': ''}
                except Exception as e:
                    return {'correct': False, 'score': 0.0, 'message': str(e)}

        # Merge options with defaults from context
        merged_options = {
            'studentsMustReduceFractions': False,
            'showFractionReductionWarnings': True,
            'requireFraction': False,
            'requireProperFraction': False,
            'strictFractions': False,
        }

        if self.context:
            # Get defaults from context flags
            ctx_flags = self.context.flags
            merged_options['studentsMustReduceFractions'] = ctx_flags.get(
                'studentsMustReduceFractions', False)
            merged_options['requireProperFraction'] = ctx_flags.get(
                'requireProperFractions', False)
            merged_options['strictFractions'] = ctx_flags.get(
                'strictFractions', False)

        # Override with provided options
        merged_options.update(options)

        return FractionAnswerChecker(self, merged_options)

    # Arithmetic operators

    def __add__(self, other: Any) -> Fraction | Real:
        """Addition: self + other."""
        if isinstance(other, Fraction):
            # a/b + c/d = (ad + bc)/(bd)
            # Use LCM for better performance
            l = lcm(self._den, other._den)
            new_num = self._num * (l // self._den) + \
                other._num * (l // other._den)
            return Fraction(new_num, l, self.context)
        elif isinstance(other, (int, float, Real)):
            value = other.value if isinstance(other, Real) else other
            other_frac = Fraction(value, 1, self.context)
            return self + other_frac
        return NotImplemented

    def __radd__(self, other: Any) -> Fraction | Real:
        """Right addition: other + self."""
        return self.__add__(other)

    def __sub__(self, other: Any) -> Fraction | Real:
        """Subtraction: self - other."""
        if isinstance(other, Fraction):
            l = lcm(self._den, other._den)
            new_num = self._num * (l // self._den) - \
                other._num * (l // other._den)
            return Fraction(new_num, l, self.context)
        elif isinstance(other, (int, float, Real)):
            value = other.value if isinstance(other, Real) else other
            other_frac = Fraction(value, 1, self.context)
            return self - other_frac
        return NotImplemented

    def __rsub__(self, other: Any) -> Fraction | Real:
        """Right subtraction: other - self."""
        if isinstance(other, (int, float, Real)):
            value = other.value if isinstance(other, Real) else other
            other_frac = Fraction(value, 1, self.context)
            return other_frac - self
        return NotImplemented

    def __mul__(self, other: Any) -> Fraction | Real:
        """Multiplication: self * other."""
        if isinstance(other, Fraction):
            # (a/b) * (c/d) = (ac)/(bd)
            return Fraction(self._num * other._num, self._den * other._den, self.context)
        elif isinstance(other, (int, float, Real)):
            value = other.value if isinstance(other, Real) else other
            other_frac = Fraction(value, 1, self.context)
            return self * other_frac
        return NotImplemented

    def __rmul__(self, other: Any) -> Fraction | Real:
        """Right multiplication: other * self."""
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Fraction | Real:
        """Division: self / other."""
        if isinstance(other, Fraction):
            # (a/b) / (c/d) = (ad)/(bc)
            return Fraction(self._num * other._den, self._den * other._num, self.context)
        elif isinstance(other, (int, float, Real)):
            value = other.value if isinstance(other, Real) else other
            other_frac = Fraction(value, 1, self.context)
            return self / other_frac
        return NotImplemented

    def __rtruediv__(self, other: Any) -> Fraction | Real:
        """Right division: other / self."""
        if isinstance(other, (int, float, Real)):
            value = other.value if isinstance(other, Real) else other
            other_frac = Fraction(value, 1, self.context)
            return other_frac / self
        return NotImplemented

    def __pow__(self, other: Any) -> Fraction | Real:
        """
        Exponentiation: self ** other.

        For fractional powers with odd denominators, can handle negative bases.

        Reference: contextFraction.pl
        """
        if isinstance(other, (int, Fraction)):
            if isinstance(other, int):
                exp = other
            else:
                # Reduce the exponent
                other_reduced = other.reduce()
                # Check if odd denominator (allows negative base)
                if self._num < 0 and other_reduced._den % 2 == 1:
                    # Can take fractional root of negative number
                    base_abs = abs(self._num)
                    result_num = int(
                        round(base_abs ** (other_reduced._num / other_reduced._den)))
                    if other_reduced._num % 2 == 1:
                        result_num = -result_num
                    result_den = int(
                        round(self._den ** (other_reduced._num / other_reduced._den)))
                    return Fraction(result_num, result_den, self.context)
                else:
                    # Standard power
                    val = (
                        self._num / self._den) ** (other_reduced._num / other_reduced._den)
                    return Real(val)

            # Integer power
            if exp >= 0:
                return Fraction(self._num ** exp, self._den ** exp, self.context)
            else:
                # Negative power: flip and negate exponent
                return Fraction(self._den ** (-exp), self._num ** (-exp), self.context)

        # Fall back to Real
        return Real((self._num / self._den) ** float(other))

    def __rpow__(self, other: Any) -> Real:
        """Right exponentiation: other ** self."""
        exp = self._num / self._den
        return Real(float(other) ** exp)

    def __neg__(self) -> Fraction:
        """Unary negation: -self."""
        return Fraction(-self._num, self._den, self.context, reduce=False)

    def __pos__(self) -> Fraction:
        """Unary positive: +self."""
        return self

    def __abs__(self) -> Fraction:
        """Absolute value: abs(self)."""
        return Fraction(abs(self._num), self._den, self.context, reduce=False)

    def __eq__(self, other: Any) -> bool:
        """Equality check."""
        if isinstance(other, Fraction):
            return self.compare(other)
        return False

    def __lt__(self, other: Any) -> bool:
        """Less than."""
        if isinstance(other, Fraction):
            return self._num * other._den < other._num * self._den
        elif isinstance(other, (int, float, Real)):
            value = other.value if isinstance(other, Real) else other
            return self._num / self._den < value
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        """Less than or equal."""
        return self == other or self < other

    def __gt__(self, other: Any) -> bool:
        """Greater than."""
        if isinstance(other, Fraction):
            return self._num * other._den > other._num * self._den
        elif isinstance(other, (int, float, Real)):
            value = other.value if isinstance(other, Real) else value
            return self._num / self._den > value
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        """Greater than or equal."""
        return self == other or self > other


def continued_fraction(x: float, max_denominator: int = 10**8) -> tuple[int, int]:
    """
    Convert a real number to a fraction using continued fractions.

    Args:
        x: Real number to convert
        max_denominator: Maximum allowed denominator

    Returns:
        Tuple of (numerator, denominator)

    Reference: contextFraction.pl::continuedFraction
    """
    step = x
    n = int(step)
    h0, h1, k0, k1 = 1, n, 0, 1

    # End when step is an integer or denominator exceeds max
    while step != n and k1 <= max_denominator:
        step = 1 / (step - n)
        n = int(step)

        # Compute next numerator and denominator
        h0, h1 = h1, n * h1 + h0
        k0, k1 = k1, n * k1 + k0

        # Check denominator limit
        if k1 > max_denominator:
            return (h0, k0)

    return (h1, k1)

