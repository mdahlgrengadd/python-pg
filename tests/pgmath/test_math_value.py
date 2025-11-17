"""Tests for MathValue base class behaviors."""

from typing import ClassVar

from pydantic import BaseModel

from pg.math.collections import List as MathList
from pg.math.numeric import Complex, Real
from pg.math.value import MathValue, TypePrecedence


class DummyValue(BaseModel, MathValue):
    """Simple concrete MathValue for testing abstract helpers."""

    type_precedence: ClassVar[TypePrecedence] = TypePrecedence.NUMBER
    value: float

    def promote(self, other: MathValue) -> MathValue:
        return DummyValue(value=float(other.to_python()))

    def compare(self, other: MathValue, tolerance: float = 0.001, mode: str = "relative") -> bool:
        if not isinstance(other, DummyValue):
            return False
        return abs(self.value - other.value) <= tolerance

    def to_string(self) -> str:
        return str(self.value)

    def to_tex(self) -> str:
        return str(self.value)

    def to_python(self) -> float:
        return self.value

    def __add__(self, other: float) -> MathValue:
        return DummyValue(self.value + float(other))

    def __radd__(self, other: float) -> MathValue:
        return self.__add__(other)

    def __sub__(self, other: float) -> MathValue:
        return DummyValue(self.value - float(other))

    def __rsub__(self, other: float) -> MathValue:
        return DummyValue(float(other) - self.value)

    def __mul__(self, other: float) -> MathValue:
        return DummyValue(self.value * float(other))

    def __rmul__(self, other: float) -> MathValue:
        return self.__mul__(other)

    def __truediv__(self, other: float) -> MathValue:
        return DummyValue(self.value / float(other))

    def __rtruediv__(self, other: float) -> MathValue:
        return DummyValue(float(other) / self.value)

    def __pow__(self, other: float) -> MathValue:
        return DummyValue(self.value ** float(other))

    def __rpow__(self, other: float) -> MathValue:
        return DummyValue(float(other) ** self.value)

    def __neg__(self) -> MathValue:
        return DummyValue(-self.value)

    def __pos__(self) -> MathValue:
        return DummyValue(+self.value)

    def __abs__(self) -> MathValue:
        return DummyValue(abs(self.value))


class TestMathValueUtilities:
    """Tests that focus on MathValue helper APIs."""

    def test_should_promote_to_uses_type_precedence(self):
        class HigherValue(DummyValue):
            type_precedence = TypePrecedence.VECTOR

        assert DummyValue.should_promote_to(HigherValue) is True
        assert HigherValue.should_promote_to(DummyValue) is False

    def test_promote_types_returns_common_type(self):
        real = Real(2)
        complex_value = Complex(1, 1)
        promoted_real, promoted_complex = real.promote_types(complex_value)
        assert isinstance(promoted_real, Complex)
        assert isinstance(promoted_complex, Complex)
        assert promoted_real.compare(Complex(2, 0))

    def test_from_python_converts_python_types(self):
        assert isinstance(MathValue.from_python(4), Real)
        math_list = MathValue.from_python([1, 2])
        assert isinstance(math_list, MathList)
        assert math_list.elements[0].value == 1.0

    def test_compare_with_tolerance(self):
        base = DummyValue(value=1.0)
        assert base.compare(DummyValue(value=1.0005), tolerance=0.001)
        assert base.compare(DummyValue(value=1.01), tolerance=0.001) is False

    def test_promote_method_example(self):
        value = DummyValue(value=2.0)
        promoted = value.promote(DummyValue(value=5.0))
        assert isinstance(promoted, DummyValue)
        assert promoted.value == 5.0

    def test_from_python_coerces_bool(self):
        real_true = MathValue.from_python(True)
        real_false = MathValue.from_python(False)
        assert isinstance(real_true, Real)
        assert real_true.value == 1.0
        assert real_false.value == 0.0
