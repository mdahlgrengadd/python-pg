"""
Collection MathValue types: List, String.

Reference: lib/Value/List.pm, lib/Value/String.pm
"""

from __future__ import annotations

from typing import Any

from .value import MathValue, ToleranceMode, TypePrecedence


class List(MathValue):
    """
    List/sequence of MathValue elements.

    Reference: lib/Value/List.pm
    """

    type_precedence = TypePrecedence.LIST

    def __init__(self, elements: list[MathValue]):
        """
        Initialize a List.

        Args:
            elements: List of MathValue elements
        """
        self.elements = elements

    def promote(self, other: MathValue) -> MathValue:
        """Lists don't promote to other types."""
        return self

    def compare(
        self, other: MathValue, tolerance: float = 0.001, mode: str = ToleranceMode.RELATIVE
    ) -> bool:
        """Compare lists element-wise."""
        if not isinstance(other, List):
            return False

        if len(self.elements) != len(other.elements):
            return False

        return all(
            el1.compare(el2, tolerance, mode)
            for el1, el2 in zip(self.elements, other.elements)
        )

    def to_string(self) -> str:
        """Convert to string."""
        def elem_to_string(el):
            if isinstance(el, str):
                return el
            elif hasattr(el, 'to_string'):
                return el.to_string()
            else:
                return str(el)
        
        elements_str = ", ".join(elem_to_string(el) for el in self.elements)
        return f"[{elements_str}]"

    def to_tex(self) -> str:
        """Convert to LaTeX."""
        def elem_to_tex(el):
            if isinstance(el, str):
                return el  # Assume string is already in appropriate format
            elif hasattr(el, 'to_tex'):
                return el.to_tex()
            else:
                return str(el)
        
        elements_str = ", ".join(elem_to_tex(el) for el in self.elements)
        return f"\\left[{elements_str}\\right]"

    def to_python(self) -> list[Any]:
        """Convert to Python list."""
        return [el.to_python() for el in self.elements]

    def __len__(self) -> int:
        """Length of list."""
        return len(self.elements)

    def __getitem__(self, index: int) -> MathValue:
        """Get element by index."""
        return self.elements[index]

    def cmp(self, *args, **kwargs) -> "MathValue":
        """
        Return a comparator for this list.

        In Perl MathObjects, cmp() returns a comparator object.
        For Python, we return self to maintain compatibility.
        """
        return self

    def __setitem__(self, index: int, value: MathValue) -> None:
        """Set element by index."""
        self.elements[index] = value

    # Arithmetic operators (element-wise for compatible types)

    def __add__(self, other: Any) -> MathValue:
        """List concatenation or element-wise addition."""
        if isinstance(other, List):
            # Concatenation
            return List(self.elements + other.elements)
        else:
            # Element-wise addition (broadcast scalar)
            from .value import MathValue

            if isinstance(other, MathValue):
                return List([el + other for el in self.elements])
            return NotImplemented

    def __radd__(self, other: Any) -> MathValue:
        """Right addition."""
        from .value import MathValue

        if isinstance(other, MathValue):
            return List([other + el for el in self.elements])
        return NotImplemented

    def __sub__(self, other: Any) -> MathValue:
        """Element-wise subtraction."""
        if isinstance(other, List):
            if len(self.elements) != len(other.elements):
                raise ValueError("List dimensions must match for subtraction")
            return List([el1 - el2 for el1, el2 in zip(self.elements, other.elements)])
        else:
            from .value import MathValue

            if isinstance(other, MathValue):
                return List([el - other for el in self.elements])
            return NotImplemented

    def __rsub__(self, other: Any) -> MathValue:
        """Right subtraction."""
        from .value import MathValue

        if isinstance(other, MathValue):
            return List([other - el for el in self.elements])
        return NotImplemented

    def __mul__(self, other: Any) -> MathValue:
        """Scalar multiplication or element-wise multiplication."""
        if isinstance(other, List):
            # Element-wise multiplication
            if len(self.elements) != len(other.elements):
                raise ValueError("List dimensions must match for multiplication")
            return List([el1 * el2 for el1, el2 in zip(self.elements, other.elements)])
        else:
            # Scalar multiplication
            from .value import MathValue

            if isinstance(other, (int, float, MathValue)):
                return List([el * other for el in self.elements])
            return NotImplemented

    def __rmul__(self, other: Any) -> MathValue:
        """Right multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> MathValue:
        """Element-wise division."""
        if isinstance(other, List):
            if len(self.elements) != len(other.elements):
                raise ValueError("List dimensions must match for division")
            return List([el1 / el2 for el1, el2 in zip(self.elements, other.elements)])
        else:
            from .value import MathValue

            if isinstance(other, (int, float, MathValue)):
                return List([el / other for el in self.elements])
            return NotImplemented

    def __rtruediv__(self, other: Any) -> MathValue:
        """Right division."""
        from .value import MathValue

        if isinstance(other, MathValue):
            return List([other / el for el in self.elements])
        return NotImplemented

    def __pow__(self, other: Any) -> MathValue:
        """Element-wise power."""
        from .value import MathValue

        if isinstance(other, (int, float, MathValue)):
            return List([el**other for el in self.elements])
        return NotImplemented

    def __rpow__(self, other: Any) -> MathValue:
        """Right power."""
        from .value import MathValue

        if isinstance(other, MathValue):
            return List([other**el for el in self.elements])
        return NotImplemented

    def __neg__(self) -> List:
        """Unary negation."""
        return List([-el for el in self.elements])

    def __pos__(self) -> List:
        """Unary positive."""
        return List([+el for el in self.elements])

    def __abs__(self) -> List:
        """Element-wise absolute value."""
        return List([abs(el) for el in self.elements])


class String(MathValue):
    """
    String value (for answer checking, labels, etc.).

    Reference: lib/Value/String.pm
    """

    type_precedence = TypePrecedence.STRING

    def __init__(self, value: str):
        """
        Initialize a String.

        Args:
            value: String value
        """
        self.value = value

    def promote(self, other: MathValue) -> MathValue:
        """Strings don't promote."""
        return self

    def compare(
        self, other: MathValue, tolerance: float = 0.001, mode: str = ToleranceMode.RELATIVE
    ) -> bool:
        """String comparison (exact match by default)."""
        if not isinstance(other, String):
            return False
        return self.value == other.value

    def to_string(self) -> str:
        """Convert to string."""
        return self.value

    def to_tex(self) -> str:
        """Convert to LaTeX."""
        return f"\\text{{{self.value}}}"

    def to_python(self) -> str:
        """Convert to Python string."""
        return self.value

    def __len__(self) -> int:
        """Length of string."""
        return len(self.value)

    def cmp(self, *args, **kwargs) -> "MathValue":
        """
        Return a comparator for this string.

        In Perl MathObjects, cmp() returns a comparator object.
        For Python, we return self to maintain compatibility.
        """
        return self

    # String operations

    def __add__(self, other: Any) -> String:
        """String concatenation."""
        if isinstance(other, String):
            return String(self.value + other.value)
        elif isinstance(other, str):
            return String(self.value + other)
        else:
            return NotImplemented  # type: ignore

    def __radd__(self, other: Any) -> String:
        """Right concatenation."""
        if isinstance(other, str):
            return String(other + self.value)
        else:
            return NotImplemented  # type: ignore

    def __mul__(self, other: Any) -> String:
        """String repetition."""
        if isinstance(other, int):
            return String(self.value * other)
        else:
            return NotImplemented  # type: ignore

    def __rmul__(self, other: Any) -> String:
        """Right repetition."""
        return self.__mul__(other)

    # Not supported for strings

    def __sub__(self, other: Any) -> MathValue:
        """Subtraction not supported."""
        raise TypeError("String does not support subtraction")

    def __rsub__(self, other: Any) -> MathValue:
        """Right subtraction not supported."""
        raise TypeError("String does not support subtraction")

    def __truediv__(self, other: Any) -> MathValue:
        """Division not supported."""
        raise TypeError("String does not support division")

    def __rtruediv__(self, other: Any) -> MathValue:
        """Right division not supported."""
        raise TypeError("String does not support division")

    def __pow__(self, other: Any) -> MathValue:
        """Power not supported."""
        raise TypeError("String does not support exponentiation")

    def __rpow__(self, other: Any) -> MathValue:
        """Right power not supported."""
        raise TypeError("String does not support exponentiation")

    def __neg__(self) -> MathValue:
        """Negation not supported."""
        raise TypeError("String does not support negation")

    def __pos__(self) -> MathValue:
        """Unary positive not supported."""
        raise TypeError("String does not support unary positive")

    def __abs__(self) -> MathValue:
        """Absolute value not supported."""
        raise TypeError("String does not support absolute value")
