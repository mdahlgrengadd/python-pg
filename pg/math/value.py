"""
Base MathValue class for the WeBWorK PG MathObjects system.

This module provides the foundation for intelligent mathematical value objects with:
- Type promotion hierarchy
- Operator overloading
- Fuzzy comparison with tolerances
- Multiple output formats (string, TeX)

Reference: lib/Value.pm (lines 1-200) in legacy Perl codebase
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, ClassVar


class TypePrecedence(IntEnum):
    """
    Type promotion precedence hierarchy.

    Lower values promote to higher values.
    Based on lib/Value.pm type precedence system.
    """

    NUMBER = 0  # Generic number
    REAL = 1  # Real number
    FRACTION = 2  # Fraction (between Real and Infinity per Perl)
    INFINITY = 3  # Infinity (special)
    COMPLEX = 4  # Complex number
    POINT = 5  # Point in n-space
    VECTOR = 6  # Vector
    MATRIX = 7  # Matrix
    LIST = 8  # List/sequence
    INTERVAL = 9  # Interval
    SET = 10  # Set
    UNION = 11  # Union of intervals/sets
    STRING = 12  # String
    FORMULA = 13  # Formula (highest - contains expressions)


class ToleranceMode:
    """Modes for fuzzy comparison."""

    RELATIVE = "relative"  # |a - b| / |b| < tol
    ABSOLUTE = "absolute"  # |a - b| < tol
    SIGFIGS = "sigfigs"  # Significant figures


class MathValue(ABC):
    """
    Base class for all mathematical value objects.

    Provides:
    - Type promotion system
    - Operator overloading (all Python operators)
    - Fuzzy comparison with tolerances
    - Multiple output formats (string, TeX, etc.)

    Subclasses must implement:
    - type_precedence: Class variable defining promotion order
    - All abstract methods

    Note: Concrete subclasses should inherit from both BaseModel and MathValue,
    e.g., `class Real(BaseModel, MathValue):`. MathValue itself is abstract
    and does not inherit from BaseModel to avoid MRO conflicts.
    """

    # Type precedence for promotion (must be set by subclasses)
    type_precedence: ClassVar[TypePrecedence]

    @abstractmethod
    def promote(self, other: MathValue) -> MathValue:
        """
        Promote this value to be compatible with another type.

        Args:
            other: The other value to promote to

        Returns:
            Promoted version of self (or self if no promotion needed)

        Example:
            Real(2).promote(Complex(1, 1)) → Complex(2, 0)
        """
        pass

    @abstractmethod
    def compare(
        self, other: MathValue, tolerance: float = 0.001, mode: str = ToleranceMode.RELATIVE
    ) -> bool:
        """
        Fuzzy comparison with tolerance.

        Args:
            other: Value to compare against
            tolerance: Tolerance for comparison
            mode: Tolerance mode (relative, absolute, sigfigs)

        Returns:
            True if values are equal within tolerance
        """
        pass

    # String representations

    @abstractmethod
    def to_string(self) -> str:
        """Convert to human-readable string."""
        pass

    @abstractmethod
    def to_tex(self) -> str:
        """Convert to LaTeX representation."""
        pass

    def __str__(self) -> str:
        """String representation (uses to_string)."""
        return self.to_string()

    def __repr__(self) -> str:
        """Debug representation."""
        return f"{self.__class__.__name__}({self.to_string()})"

    # Operator overloading (Python magic methods)

    @abstractmethod
    def __add__(self, other: Any) -> MathValue:
        """Addition: self + other"""
        pass

    @abstractmethod
    def __radd__(self, other: Any) -> MathValue:
        """Right addition: other + self"""
        pass

    @abstractmethod
    def __sub__(self, other: Any) -> MathValue:
        """Subtraction: self - other"""
        pass

    @abstractmethod
    def __rsub__(self, other: Any) -> MathValue:
        """Right subtraction: other - self"""
        pass

    @abstractmethod
    def __mul__(self, other: Any) -> MathValue:
        """Multiplication: self * other"""
        pass

    @abstractmethod
    def __rmul__(self, other: Any) -> MathValue:
        """Right multiplication: other * self"""
        pass

    @abstractmethod
    def __truediv__(self, other: Any) -> MathValue:
        """Division: self / other"""
        pass

    @abstractmethod
    def __rtruediv__(self, other: Any) -> MathValue:
        """Right division: other / self"""
        pass

    @abstractmethod
    def __pow__(self, other: Any) -> MathValue:
        """Exponentiation: self ** other"""
        pass

    @abstractmethod
    def __rpow__(self, other: Any) -> MathValue:
        """Right exponentiation: other ** self"""
        pass

    @abstractmethod
    def __neg__(self) -> MathValue:
        """Unary negation: -self"""
        pass

    @abstractmethod
    def __pos__(self) -> MathValue:
        """Unary positive: +self"""
        pass

    @abstractmethod
    def __abs__(self) -> MathValue:
        """Absolute value: abs(self)"""
        pass

    # Comparison operators (using fuzzy comparison)

    def __eq__(self, other: Any) -> bool:
        """Equality with default tolerance."""
        if not isinstance(other, MathValue):
            # Try to convert to MathValue
            from .numeric import Real

            try:
                other = Real(float(other))
            except (TypeError, ValueError):
                return False

        return self.compare(other)

    def __ne__(self, other: Any) -> bool:
        """Inequality."""
        return not self.__eq__(other)

    # Note: <, >, <=, >= need to be implemented by subclasses where applicable
    # (e.g., Real supports ordering, but Complex does not)

    # Configuration methods (Perl-style chainable methods)

    def with_params(self, **kwargs) -> MathValue:
        """
        Configure properties of this MathValue.

        This is the Python equivalent of Perl MathObjects' .with() method.
        It allows configuration of answer checking properties like tolerance, period, etc.

        Args:
            **kwargs: Configuration options (period, tolerance, etc.)

        Returns:
            self (for method chaining)

        Example:
            Real('pi / 2').with_params(period=pi)  # Accepts pi/2 + n*pi for any integer n
        """
        # Store configuration in a private attribute for answer checking to use
        if not hasattr(self, '_with_config'):
            self._with_config = {}
        self._with_config.update(kwargs)
        return self

    # Helper methods for type promotion

    @classmethod
    def should_promote_to(cls, other_type: type[MathValue]) -> bool:
        """
        Check if this type should promote to another type.

        Args:
            other_type: The type to compare against

        Returns:
            True if this type should promote to other_type
        """
        return cls.type_precedence < other_type.type_precedence

    def promote_types(self, other: MathValue) -> tuple[MathValue, MathValue]:
        """
        Promote both values to a common type.

        Args:
            other: The other value

        Returns:
            Tuple of (promoted_self, promoted_other)

        Example:
            Real(2).promote_types(Complex(1, 1)) → (Complex(2, 0), Complex(1, 1))
        """
        if self.type_precedence < other.type_precedence:
            # Promote self to other's type
            return self.promote(other), other
        elif other.type_precedence < self.type_precedence:
            # Promote other to self's type
            return self, other.promote(self)
        else:
            # Same type, no promotion needed
            return self, other

    # Conversion helpers

    @classmethod
    def from_python(cls, value: Any) -> MathValue:
        """
        Convert a Python value to a MathValue.

        This is a factory method that dispatches to the appropriate subclass.

        Args:
            value: Python value (int, float, complex, list, etc.)

        Returns:
            Appropriate MathValue subclass instance
        """
        # Import here to avoid circular imports
        from .collections import List as MathList
        from .collections import String as MathString
        from .geometric import Matrix, Point, Vector
        from .numeric import Complex as MathComplex
        from .numeric import Infinity, Real

        if isinstance(value, MathValue):
            return value

        elif isinstance(value, bool):
            # bool is a subclass of int, so check first
            return Real(1.0 if value else 0.0)

        elif isinstance(value, int):
            return Real(float(value))

        elif isinstance(value, float):
            if value == float("inf"):
                return Infinity(1)
            elif value == float("-inf"):
                return Infinity(-1)
            else:
                return Real(value)

        elif isinstance(value, complex):
            return MathComplex(value.real, value.imag)

        elif isinstance(value, str):
            return MathString(value)

        elif isinstance(value, (list, tuple)):
            # Could be Point, Vector, Matrix, or List
            if len(value) > 0:
                # Check if it's a matrix (list of lists)
                if isinstance(value[0], (list, tuple)):
                    # Matrix
                    rows = [[cls.from_python(el) for el in row] for row in value]
                    return Matrix(rows)
                else:
                    # Could be Point, Vector, or List
                    # Default to List for now (can be specialized in context)
                    elements = [cls.from_python(el) for el in value]
                    return MathList(elements)
            else:
                return MathList([])

        else:
            raise TypeError(f"Cannot convert {type(value)} to MathValue")

    def to_python(self) -> Any:
        """
        Convert MathValue to Python native type.

        Returns:
            Python native value (int, float, complex, list, etc.)
        """
        # Default implementation - subclasses should override
        raise NotImplementedError(f"{self.__class__.__name__}.to_python() not implemented")
