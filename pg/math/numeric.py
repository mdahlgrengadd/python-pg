"""
Numeric MathValue types: Real, Complex, Infinity.

These are the fundamental numeric types in the MathObjects system.

Reference: lib/Value/Real.pm, lib/Value/Complex.pm, lib/Value/Infinity.pm
"""

from __future__ import annotations

import ast
import math
from typing import Any

from pydantic import BaseModel, Field, field_validator, ConfigDict

from .value import MathValue, ToleranceMode, TypePrecedence


# Safe namespace with allowed math functions and constants (defined at module level)
_SAFE_MATH_NAMESPACE = {
    'pi': math.pi,
    'e': math.e,
    'sqrt': math.sqrt,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'asin': math.asin,
    'acos': math.acos,
    'atan': math.atan,
    'exp': math.exp,
    'log': math.log,
    'log10': math.log10,
    'abs': abs,
    '__builtins__': {},
}


def _safe_eval_math_expression(expr: str) -> float:
    """
    Safely evaluate a mathematical expression containing only math functions and constants.

    This replaces the unsafe eval() call with AST-based evaluation.
    Supports: math constants (pi, e), math functions (sqrt, sin, cos, tan, etc),
    and basic arithmetic operators (+, -, *, /, **)

    Args:
        expr: Mathematical expression string (e.g., "pi/2", "sqrt(4)")

    Returns:
        float: Evaluated result

    Raises:
        ValueError: If expression is invalid or contains unsafe operations
    """
    try:
        # Parse the expression into an AST
        node = ast.parse(expr, mode='eval')

        # Validate that only safe operations are used
        for child in ast.walk(node):
            # Only allow these node types
            allowed_types = (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Call,
                ast.Name,
                ast.Constant,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.Pow,
                ast.UAdd,
                ast.USub,
                ast.Load,  # Context node for variable loading
            )
            if not isinstance(child, allowed_types):
                raise ValueError(f"Unsafe operation in expression: {expr}")

            # Validate that Name nodes only reference allowed identifiers
            if isinstance(child, ast.Name):
                if child.id not in _SAFE_MATH_NAMESPACE:
                    raise ValueError(f"Undefined name '{child.id}' in expression: {expr}")

        # Compile and evaluate the AST
        code = compile(node, '<string>', 'eval')
        result = eval(code, _SAFE_MATH_NAMESPACE)
        return float(result)

    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {expr}") from e
    except Exception as e:
        raise ValueError(f"Could not evaluate expression '{expr}': {e}") from e


class Real(BaseModel, MathValue):
    """
    Real number value.

    The most common mathematical type, represents floating-point numbers
    with fuzzy comparison support.

    Reference: lib/Value/Real.pm
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: float = Field(description="The numeric value")
    context: Any = Field(default=None, description="The evaluation context")
    type_precedence: TypePrecedence = Field(default=TypePrecedence.REAL, init=False)

    def __init__(self, value: float | int | str, context=None, **kwargs):
        """
        Initialize a Real number.

        Args:
            value: Numeric value (will be converted to float) or string expression (e.g., 'pi', 'pi/2')
            context: The Context (None = use current default)
        """
        # Handle string expressions like 'pi', 'pi/2', etc.
        if isinstance(value, str):
            try:
                # Try to evaluate as a simple number first
                float_value = float(value)
            except (ValueError, SyntaxError):
                # Try to evaluate as a mathematical expression using safe parser
                float_value = _safe_eval_math_expression(value)
        else:
            float_value = float(value)

        if context is None:
            # Import here to avoid circular dependency
            from .context import get_current_context
            context = get_current_context()

        super().__init__(value=float_value, context=context, **kwargs)

    def promote(self, other: MathValue) -> MathValue:
        """Promote Real to another type."""
        if isinstance(other, Complex):
            return Complex(self.value, 0.0, self.context)
        elif isinstance(other, Infinity):
            # Real doesn't promote to Infinity
            return self
        else:
            # For Point, Vector, etc., promotion happens at that level
            return self

    def compare(
        self, other: MathValue, tolerance: float = 0.001, mode: str = ToleranceMode.RELATIVE
    ) -> bool:
        """Fuzzy comparison of real numbers."""
        # Promote types if needed
        if not isinstance(other, Real):
            self_promoted, other_promoted = self.promote_types(other)
            if self_promoted is not self:
                return self_promoted.compare(other_promoted, tolerance, mode)
            # Can't promote, not comparable
            return False

        return fuzzy_compare(self.value, other.value, tolerance, mode)

    def __eq__(self, other: Any) -> bool:
        """
        Equality comparison with context-aware tolerance.

        Uses tolerance and tolType from context for fuzzy comparison.
        Reference: lib/Value/Real.pm lines 138-177
        """
        # Get other value
        if isinstance(other, (int, float)):
            other_value = float(other)
        elif isinstance(other, Real):
            other_value = other.value
        else:
            return False

        # Get tolerance settings from context
        tolerance = self.context.flags.get('tolerance')
        if tolerance is None:
            tolerance = 0.001  # Default tolerance

        tol_type = self.context.flags.get('tolType')
        if tol_type is None:
            tol_type = 'relative'  # Default to relative

        zero_level = self.context.flags.get('zeroLevel')
        if zero_level is None:
            zero_level = 1e-14  # Default zero level

        zero_level_tol = self.context.flags.get('zeroLevelTol')
        if zero_level_tol is None:
            zero_level_tol = 1e-12  # Default zero level tolerance

        # Exact equality
        if self.value == other_value:
            return True

        # Use epsilon for floating point comparisons
        EPSILON = 1e-12

        if tol_type == 'absolute':
            # Absolute tolerance
            return abs(self.value - other_value) <= tolerance + EPSILON

        elif tol_type == 'relative':
            # Relative tolerance with special handling near zero
            # Reference: pg_mathobjects/real.py lines 135-137
            # If self is near zero, check if other is within tolerance of zero
            if abs(self.value) < zero_level:
                return abs(other_value) < tolerance

            # Otherwise use relative tolerance
            return abs(self.value - other_value) / abs(self.value) < tolerance

        else:
            # Unknown tolerance type, fall back to relative
            max_abs = max(abs(self.value), abs(other_value))
            if max_abs == 0:
                return abs(self.value - other_value) <= tolerance + EPSILON
            return abs(self.value - other_value) / max_abs <= tolerance + EPSILON

    def __ne__(self, other: Any) -> bool:
        """Not equal."""
        return not self.__eq__(other)

    def to_string(self) -> str:
        """Convert to string."""
        # Remove .0 for integers
        if self.value == int(self.value) and abs(self.value) < 1e10:
            return str(int(self.value))
        return str(self.value)

    def to_tex(self) -> str:
        """Convert to LaTeX."""
        return self.to_string()

    def to_python(self) -> float:
        """Convert to Python float."""
        return self.value

    def __str__(self) -> str:
        """String representation (for str() builtin)."""
        return self.to_string()

    def __float__(self) -> float:
        """Convert to Python float (for float() builtin)."""
        return self.value

    # Arithmetic operators

    def __add__(self, other: Any) -> MathValue:
        """Addition."""
        if isinstance(other, (int, float)):
            return Real(self.value + other, self.context)
        elif isinstance(other, Real):
            return Real(self.value + other.value, self.context)
        elif isinstance(other, MathValue):
            # Promote and retry
            self_promoted, other_promoted = self.promote_types(other)
            if self_promoted is not self:
                return self_promoted + other_promoted
            raise TypeError(f"Cannot add Real and {type(other)}")
        else:
            return NotImplemented

    def __radd__(self, other: Any) -> MathValue:
        """Right addition."""
        return self.__add__(other)

    def __sub__(self, other: Any) -> MathValue:
        """Subtraction."""
        if isinstance(other, (int, float)):
            return Real(self.value - other, self.context)
        elif isinstance(other, Real):
            return Real(self.value - other.value, self.context)
        elif isinstance(other, MathValue):
            self_promoted, other_promoted = self.promote_types(other)
            if self_promoted is not self:
                return self_promoted - other_promoted
            raise TypeError(f"Cannot subtract {type(other)} from Real")
        else:
            return NotImplemented

    def __rsub__(self, other: Any) -> MathValue:
        """Right subtraction."""
        if isinstance(other, (int, float)):
            return Real(other - self.value, self.context)
        else:
            return NotImplemented

    def __mul__(self, other: Any) -> MathValue:
        """Multiplication."""
        if isinstance(other, (int, float)):
            return Real(self.value * other, self.context)
        elif isinstance(other, Real):
            return Real(self.value * other.value, self.context)
        elif isinstance(other, MathValue):
            self_promoted, other_promoted = self.promote_types(other)
            if self_promoted is not self:
                return self_promoted * other_promoted
            raise TypeError(f"Cannot multiply Real and {type(other)}")
        else:
            return NotImplemented

    def __rmul__(self, other: Any) -> MathValue:
        """Right multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> MathValue:
        """Division."""
        if isinstance(other, (int, float)):
            if other == 0:
                # Division by zero -> infinity
                return Infinity(1 if self.value > 0 else -1 if self.value < 0 else 0, self.context)
            return Real(self.value / other, self.context)
        elif isinstance(other, Real):
            if other.value == 0:
                return Infinity(1 if self.value > 0 else -1 if self.value < 0 else 0, self.context)
            return Real(self.value / other.value, self.context)
        elif isinstance(other, MathValue):
            self_promoted, other_promoted = self.promote_types(other)
            if self_promoted is not self:
                return self_promoted / other_promoted
            raise TypeError(f"Cannot divide Real by {type(other)}")
        else:
            return NotImplemented

    def __rtruediv__(self, other: Any) -> MathValue:
        """Right division."""
        if isinstance(other, (int, float)):
            if self.value == 0:
                return Infinity(1 if other > 0 else -1 if other < 0 else 0, self.context)
            return Real(other / self.value, self.context)
        else:
            return NotImplemented

    def __pow__(self, other: Any) -> MathValue:
        """Exponentiation."""
        if isinstance(other, (int, float)):
            result = self.value**other
            # Check if result is complex (e.g., (-1)^0.5)
            if isinstance(result, complex):
                return Complex(result.real, result.imag, self.context)
            return Real(result, self.context)
        elif isinstance(other, Real):
            result = self.value ** other.value
            if isinstance(result, complex):
                return Complex(result.real, result.imag, self.context)
            return Real(result, self.context)
        elif isinstance(other, MathValue):
            self_promoted, other_promoted = self.promote_types(other)
            if self_promoted is not self:
                return self_promoted**other_promoted
            raise TypeError(f"Cannot raise Real to {type(other)}")
        else:
            return NotImplemented

    def __rpow__(self, other: Any) -> MathValue:
        """Right exponentiation."""
        if isinstance(other, (int, float)):
            result = other**self.value
            if isinstance(result, complex):
                return Complex(result.real, result.imag, self.context)
            return Real(result, self.context)
        else:
            return NotImplemented

    def __neg__(self) -> Real:
        """Unary negation."""
        return Real(-self.value, self.context)

    def __pos__(self) -> Real:
        """Unary positive."""
        return Real(self.value, self.context)

    def __abs__(self) -> Real:
        """Absolute value."""
        return Real(abs(self.value), self.context)

    # Comparison operators (with tolerance)

    def __lt__(self, other: Any) -> bool:
        """Less than."""
        if isinstance(other, (int, float)):
            return self.value < other
        elif isinstance(other, Real):
            return self.value < other.value
        else:
            return NotImplemented

    def __le__(self, other: Any) -> bool:
        """Less than or equal."""
        return self.__eq__(other) or self.__lt__(other)

    def __gt__(self, other: Any) -> bool:
        """Greater than."""
        if isinstance(other, (int, float)):
            return self.value > other
        elif isinstance(other, Real):
            return self.value > other.value
        else:
            return NotImplemented

    def __ge__(self, other: Any) -> bool:
        """Greater than or equal."""
        return self.__eq__(other) or self.__gt__(other)

    def answer_checker(self, **options):
        """
        Create an answer checker for this Real number.

        Args:
            **options: Checker options (tolerance, tolType)

        Returns:
            RealAnswerChecker that can check student answers
        """
        from .answer_checker import RealAnswerChecker
        return RealAnswerChecker(self, **options)

    def cmp(self, **options):
        """
        Alias for answer_checker() - Perl compatibility.

        Args:
            **options: Checker options (tolerance, tolType)

        Returns:
            RealAnswerChecker that can check student answers
        """
        return self.answer_checker(**options)


class Complex(MathValue):
    """
    Complex number value.

    Represents numbers with real and imaginary parts.

    Reference: lib/Value/Complex.pm
    """

    type_precedence = TypePrecedence.COMPLEX

    def __init__(self, real: float | int, imag: float | int = 0.0, context=None):
        """
        Initialize a Complex number.

        Args:
            real: Real part (can be a number, list/tuple [real, imag], or string "a+bi" for Perl compatibility)
            imag: Imaginary part (default 0)
            context: The Context (None = use current default)
        """
        # Handle list/tuple arguments (Perl compatibility): Complex([a, b])
        if isinstance(real, (list, tuple)):
            if len(real) >= 2:
                self.real = float(real[0])
                self.imag = float(real[1])
            elif len(real) == 1:
                self.real = float(real[0])
                self.imag = 0.0
            else:
                self.real = 0.0
                self.imag = 0.0
        elif isinstance(real, str):
            # Parse string like "2-4i" or "2+4i"
            import re
            match = re.match(r'([+-]?\d+(?:\.\d+)?)\s*([+-])\s*(\d+(?:\.\d+)?)i', real.replace(' ', ''))
            if match:
                self.real = float(match.group(1))
                sign = match.group(2)
                self.imag = float(match.group(3))
                if sign == '-':
                    self.imag = -self.imag
            else:
                # Try to parse as just real part
                try:
                    self.real = float(real)
                    self.imag = 0.0
                except ValueError:
                    raise ValueError(f"Cannot parse complex number from string: {real}")
        else:
            self.real = float(real)
            self.imag = float(imag)
        if context is not None:
            self.context = context
        else:
            from .context import get_current_context
            self.context = get_current_context()

    def promote(self, other: MathValue) -> MathValue:
        """Complex is high in hierarchy, doesn't promote to much."""
        # Complex doesn't promote to Point, Vector, etc.
        return self

    def compare(
        self, other: MathValue, tolerance: float = 0.001, mode: str = ToleranceMode.RELATIVE
    ) -> bool:
        """Fuzzy comparison of complex numbers."""
        if isinstance(other, Real):
            # Promote Real to Complex
            other = Complex(other.value, 0.0, self.context)

        if not isinstance(other, Complex):
            return False

        # Compare both real and imaginary parts
        return fuzzy_compare(
            self.real, other.real, tolerance, mode
        ) and fuzzy_compare(self.imag, other.imag, tolerance, mode)

    def to_string(self) -> str:
        """Convert to string."""
        if self.imag == 0:
            return Real(self.real).to_string()
        elif self.real == 0:
            if self.imag == 1:
                return "i"
            elif self.imag == -1:
                return "-i"
            else:
                return f"{Real(self.imag).to_string()}i"
        else:
            imag_str = Real(abs(self.imag)).to_string()
            if abs(self.imag) == 1:
                imag_str = ""
            sign = "+" if self.imag > 0 else "-"
            return f"{Real(self.real).to_string()} {sign} {imag_str}i"

    def to_tex(self) -> str:
        """Convert to LaTeX."""
        # Similar to string, but with proper formatting
        return self.to_string()

    def to_python(self) -> complex:
        """Convert to Python complex."""
        return complex(self.real, self.imag)

    @property
    def value(self) -> tuple[float, float]:
        """
        Get the value as a tuple (real, imag).

        Perl compatibility property - returns the components for unpacking.

        Returns:
            Tuple of (real, imag) values
        """
        return (self.real, self.imag)

    # Arithmetic operators

    def __add__(self, other: Any) -> MathValue:
        """Addition."""
        if isinstance(other, (int, float)):
            return Complex(self.real + other, self.imag, self.context)
        elif isinstance(other, Real):
            return Complex(self.real + other.value, self.imag, self.context)
        elif isinstance(other, Complex):
            return Complex(self.real + other.real, self.imag + other.imag, self.context)
        else:
            return NotImplemented

    def __radd__(self, other: Any) -> MathValue:
        """Right addition."""
        return self.__add__(other)

    def __sub__(self, other: Any) -> MathValue:
        """Subtraction."""
        if isinstance(other, (int, float)):
            return Complex(self.real - other, self.imag, self.context)
        elif isinstance(other, Real):
            return Complex(self.real - other.value, self.imag, self.context)
        elif isinstance(other, Complex):
            return Complex(self.real - other.real, self.imag - other.imag, self.context)
        else:
            return NotImplemented

    def __rsub__(self, other: Any) -> MathValue:
        """Right subtraction."""
        if isinstance(other, (int, float)):
            return Complex(other - self.real, -self.imag, self.context)
        elif isinstance(other, Real):
            return Complex(other.value - self.real, -self.imag, self.context)
        else:
            return NotImplemented

    def __mul__(self, other: Any) -> MathValue:
        """Multiplication."""
        if isinstance(other, (int, float)):
            return Complex(self.real * other, self.imag * other, self.context)
        elif isinstance(other, Real):
            return Complex(self.real * other.value, self.imag * other.value, self.context)
        elif isinstance(other, Complex):
            # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            real_part = self.real * other.real - self.imag * other.imag
            imag_part = self.real * other.imag + self.imag * other.real
            return Complex(real_part, imag_part, self.context)
        else:
            return NotImplemented

    def __rmul__(self, other: Any) -> MathValue:
        """Right multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> MathValue:
        """Division."""
        if isinstance(other, (int, float)):
            if other == 0:
                # Division by zero
                raise ZeroDivisionError("Complex division by zero")
            return Complex(self.real / other, self.imag / other, self.context)
        elif isinstance(other, Real):
            if other.value == 0:
                raise ZeroDivisionError("Complex division by zero")
            return Complex(self.real / other.value, self.imag / other.value, self.context)
        elif isinstance(other, Complex):
            # (a + bi) / (c + di) = [(a + bi)(c - di)] / (c^2 + d^2)
            denom = other.real**2 + other.imag**2
            if denom == 0:
                raise ZeroDivisionError("Complex division by zero")
            real_part = (self.real * other.real +
                         self.imag * other.imag) / denom
            imag_part = (self.imag * other.real -
                         self.real * other.imag) / denom
            return Complex(real_part, imag_part, self.context)
        else:
            return NotImplemented

    def __rtruediv__(self, other: Any) -> MathValue:
        """Right division."""
        if isinstance(other, (int, float)):
            return Complex(other, 0.0, self.context) / self
        elif isinstance(other, Real):
            return Complex(other.value, 0.0, self.context) / self
        else:
            return NotImplemented

    def __pow__(self, other: Any) -> MathValue:
        """Exponentiation (using Python's complex power)."""
        if isinstance(other, (int, float)):
            result = complex(self.real, self.imag) ** other
            return Complex(result.real, result.imag, self.context)
        elif isinstance(other, Real):
            result = complex(self.real, self.imag) ** other.value
            return Complex(result.real, result.imag, self.context)
        elif isinstance(other, Complex):
            result = complex(
                self.real, self.imag) ** complex(other.real, other.imag)
            return Complex(result.real, result.imag, self.context)
        else:
            return NotImplemented

    def __rpow__(self, other: Any) -> MathValue:
        """Right exponentiation."""
        if isinstance(other, (int, float)):
            result = other ** complex(self.real, self.imag)
            return Complex(result.real, result.imag, self.context)
        elif isinstance(other, Real):
            result = other.value ** complex(self.real, self.imag)
            return Complex(result.real, result.imag, self.context)
        else:
            return NotImplemented

    def __neg__(self) -> Complex:
        """Unary negation."""
        return Complex(-self.real, -self.imag, self.context)

    def __pos__(self) -> Complex:
        """Unary positive."""
        return Complex(self.real, self.imag, self.context)

    def __abs__(self) -> Real:
        """Absolute value (magnitude)."""
        return Real(math.sqrt(self.real**2 + self.imag**2), self.context)

    def norm(self) -> Real:
        """
        Compute the norm (magnitude) of the complex number.

        Perl compatibility method - returns the same as abs().

        Returns:
            Real: The magnitude of the complex number
        """
        return self.__abs__()

    def unit(self) -> Complex:
        """
        Compute the unit complex number (normalized to magnitude 1).

        Perl compatibility method - returns the complex number divided by its magnitude.

        Returns:
            Complex: The unit complex number in the same direction

        Raises:
            ZeroDivisionError: If the magnitude is zero
        """
        magnitude = self.__abs__()
        if magnitude.value == 0:
            raise ZeroDivisionError("Cannot compute unit of zero complex number")
        return Complex(self.real / magnitude.value, self.imag / magnitude.value, self.context)


class Infinity(MathValue):
    """
    Infinity value.

    Represents positive infinity, negative infinity, or undefined (0*inf).

    Reference: lib/Value/Infinity.pm
    """

    type_precedence = TypePrecedence.INFINITY

    def __init__(self, sign: int = 1, context=None):
        """
        Initialize Infinity.

        Args:
            sign: 1 for +inf, -1 for -inf, 0 for undefined
            context: The Context (None = use current default)
        """
        if sign > 0:
            self.sign = 1
        elif sign < 0:
            self.sign = -1
        else:
            self.sign = 0

        if context is not None:
            self.context = context
        else:
            from .context import get_current_context
            self.context = get_current_context()

    def promote(self, other: MathValue) -> MathValue:
        """Infinity doesn't promote."""
        return self

    def compare(
        self, other: MathValue, tolerance: float = 0.001, mode: str = ToleranceMode.RELATIVE
    ) -> bool:
        """Infinity comparison."""
        if isinstance(other, Infinity):
            return self.sign == other.sign
        return False

    def to_string(self) -> str:
        """Convert to string."""
        if self.sign == 1:
            return "inf"
        elif self.sign == -1:
            return "-inf"
        else:
            return "NaN"

    def to_tex(self) -> str:
        """Convert to LaTeX."""
        if self.sign == 1:
            return r"\infty"
        elif self.sign == -1:
            return r"-\infty"
        else:
            return r"\text{NaN}"

    def to_python(self) -> float:
        """Convert to Python float."""
        if self.sign == 1:
            return float("inf")
        elif self.sign == -1:
            return float("-inf")
        else:
            return float("nan")

    # Arithmetic operators

    def __add__(self, other: Any) -> MathValue:
        """Addition with infinity."""
        if isinstance(other, (int, float, Real)):
            return self
        elif isinstance(other, Infinity):
            if self.sign == other.sign:
                return self
            else:
                # inf + (-inf) = undefined
                return Infinity(0, self.context)
        else:
            return NotImplemented

    def __radd__(self, other: Any) -> MathValue:
        """Right addition."""
        return self.__add__(other)

    def __sub__(self, other: Any) -> MathValue:
        """Subtraction."""
        if isinstance(other, (int, float, Real)):
            return self
        elif isinstance(other, Infinity):
            if self.sign == -other.sign:
                return self
            else:
                # inf - inf = undefined
                return Infinity(0, self.context)
        else:
            return NotImplemented

    def __rsub__(self, other: Any) -> MathValue:
        """Right subtraction."""
        return -self

    def __mul__(self, other: Any) -> MathValue:
        """Multiplication."""
        if isinstance(other, (int, float)):
            if other == 0:
                return Infinity(0, self.context)  # 0 * inf = undefined
            return Infinity(self.sign * (1 if other > 0 else -1), self.context)
        elif isinstance(other, Real):
            if other.value == 0:
                return Infinity(0, self.context)
            return Infinity(self.sign * (1 if other.value > 0 else -1), self.context)
        elif isinstance(other, Infinity):
            return Infinity(self.sign * other.sign, self.context)
        else:
            return NotImplemented

    def __rmul__(self, other: Any) -> MathValue:
        """Right multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> MathValue:
        """Division."""
        if isinstance(other, (int, float, Real)):
            return self
        elif isinstance(other, Infinity):
            return Infinity(0, self.context)  # inf / inf = undefined
        else:
            return NotImplemented

    def __rtruediv__(self, other: Any) -> MathValue:
        """Right division."""
        # n / inf = 0
        return Real(0.0, self.context)

    def __pow__(self, other: Any) -> MathValue:
        """Exponentiation."""
        if isinstance(other, (int, float, Real)):
            # inf^positive = inf, inf^negative = 0, inf^0 = undefined
            if isinstance(other, Real):
                exp = other.value
            else:
                exp = other

            if exp > 0:
                return self
            elif exp < 0:
                return Real(0.0, self.context)
            else:
                return Infinity(0, self.context)  # undefined
        else:
            return NotImplemented

    def __rpow__(self, other: Any) -> MathValue:
        """Right exponentiation."""
        # base^inf
        if isinstance(other, (int, float, Real)):
            base = other.value if isinstance(other, Real) else other
            if abs(base) > 1:
                return self
            elif abs(base) < 1:
                return Real(0.0, self.context)
            else:
                return Infinity(0, self.context)  # undefined
        else:
            return NotImplemented

    def __neg__(self) -> Infinity:
        """Unary negation."""
        return Infinity(-self.sign, self.context)

    def __pos__(self) -> Infinity:
        """Unary positive."""
        return self

    def __abs__(self) -> Infinity:
        """Absolute value."""
        if self.sign == 0:
            return self
        return Infinity(1, self.context)


# Helper function for fuzzy comparison


def fuzzy_compare(a: float, b: float, tolerance: float, mode: str) -> bool:
    """
    Compare two floats with tolerance.

    Args:
        a: First value
        b: Second value
        tolerance: Tolerance value
        mode: Comparison mode (relative, absolute, sigfigs)

    Returns:
        True if values are equal within tolerance
    """
    # Exact equality
    if a == b:
        return True

    # Use epsilon for floating point comparisons to avoid precision issues
    EPSILON = 1e-12

    if mode == ToleranceMode.ABSOLUTE:
        return abs(a - b) <= tolerance + EPSILON

    elif mode == ToleranceMode.RELATIVE:
        # Avoid division by zero - use larger magnitude for relative comparison
        # PG uses: abs(a-b) / max(abs(a), abs(b)) <= tolerance
        max_abs = max(abs(a), abs(b))
        if max_abs == 0:
            return abs(a - b) <= tolerance + EPSILON
        return abs(a - b) / max_abs <= tolerance + EPSILON

    elif mode == ToleranceMode.SIGFIGS:
        # Significant figures mode
        if a == b:
            return True
        diff = abs(a - b)
        if diff == 0:
            return True
        avg = (abs(a) + abs(b)) / 2
        if avg == 0:
            return diff < 10 ** (-tolerance)
        return math.floor(math.log10(diff / avg)) < -tolerance

    else:
        raise ValueError(f"Unknown tolerance mode: {mode}")
