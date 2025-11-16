"""Tests for Infinity Pydantic model."""

import pytest
from pg.math.numeric import Infinity


class TestInfinityBasicInstantiation:
    """Test basic instantiation and sign validation."""

    def test_default_infinity(self):
        """Test creating Infinity with default (positive)."""
        inf = Infinity()
        assert inf.sign == 1
        assert inf.to_python() == float("inf")

    def test_positive_infinity(self):
        """Test creating positive infinity explicitly."""
        inf = Infinity(1)
        assert inf.sign == 1

    def test_negative_infinity(self):
        """Test creating negative infinity."""
        inf = Infinity(-1)
        assert inf.sign == -1
        assert inf.to_python() == float("-inf")

    def test_undefined_infinity(self):
        """Test creating undefined (0*inf)."""
        inf = Infinity(0)
        assert inf.sign == 0
        assert str(inf.to_python()) == "nan"

    def test_sign_normalization_positive(self):
        """Test that positive values normalize to 1."""
        inf = Infinity(5)
        assert inf.sign == 1

    def test_sign_normalization_negative(self):
        """Test that negative values normalize to -1."""
        inf = Infinity(-5)
        assert inf.sign == -1


class TestInfinityStringRepresentation:
    """Test string and TeX representations."""

    def test_positive_to_string(self):
        """Test string representation of positive infinity."""
        assert Infinity(1).to_string() == "inf"

    def test_negative_to_string(self):
        """Test string representation of negative infinity."""
        assert Infinity(-1).to_string() == "-inf"

    def test_undefined_to_string(self):
        """Test string representation of undefined."""
        assert Infinity(0).to_string() == "NaN"

    def test_positive_to_tex(self):
        """Test LaTeX representation of positive infinity."""
        assert Infinity(1).to_tex() == r"\infty"

    def test_negative_to_tex(self):
        """Test LaTeX representation of negative infinity."""
        assert Infinity(-1).to_tex() == r"-\infty"

    def test_undefined_to_tex(self):
        """Test LaTeX representation of undefined."""
        assert Infinity(0).to_tex() == r"\text{NaN}"

    def test_repr(self):
        """Test __repr__ output."""
        inf = Infinity(1)
        repr_str = repr(inf)
        assert "Infinity" in repr_str


class TestInfinityComparison:
    """Test comparison operations."""

    def test_equal_positive(self):
        """Test that two positive infinities are equal."""
        assert Infinity(1).compare(Infinity(1))

    def test_equal_negative(self):
        """Test that two negative infinities are equal."""
        assert Infinity(-1).compare(Infinity(-1))

    def test_not_equal_opposite_signs(self):
        """Test that opposite infinities are not equal."""
        assert not Infinity(1).compare(Infinity(-1))

    def test_not_equal_with_real(self):
        """Test that infinity is not equal to real numbers."""
        from pg.math.numeric import Real
        assert not Infinity(1).compare(Real(5))

    def test_not_equal_undefined(self):
        """Test that undefined is not equal to positive."""
        assert not Infinity(0).compare(Infinity(1))


class TestInfinityArithmetic:
    """Test arithmetic operations with infinity."""

    def test_add_real_number(self):
        """Test adding real number to infinity."""
        result = Infinity(1) + 5
        assert isinstance(result, Infinity)
        assert result.sign == 1

    def test_add_same_sign_infinity(self):
        """Test adding same sign infinities."""
        result = Infinity(1) + Infinity(1)
        assert isinstance(result, Infinity)
        assert result.sign == 1

    def test_add_opposite_sign_infinity(self):
        """Test adding opposite sign infinities (undefined)."""
        result = Infinity(1) + Infinity(-1)
        assert isinstance(result, Infinity)
        assert result.sign == 0

    def test_radd_real_number(self):
        """Test right addition with real number."""
        result = 5 + Infinity(1)
        assert isinstance(result, Infinity)
        assert result.sign == 1

    def test_subtract_real_number(self):
        """Test subtracting real number from infinity."""
        result = Infinity(1) - 5
        assert isinstance(result, Infinity)
        assert result.sign == 1

    def test_subtract_same_infinity(self):
        """Test subtracting same infinity (undefined)."""
        result = Infinity(1) - Infinity(1)
        assert isinstance(result, Infinity)
        assert result.sign == 0

    def test_subtract_opposite_infinity(self):
        """Test subtracting opposite infinity."""
        result = Infinity(1) - Infinity(-1)
        assert isinstance(result, Infinity)
        assert result.sign == 1

    def test_multiply_positive_number(self):
        """Test multiplying by positive number."""
        result = Infinity(1) * 5
        assert isinstance(result, Infinity)
        assert result.sign == 1

    def test_multiply_negative_number(self):
        """Test multiplying by negative number."""
        result = Infinity(1) * (-5)
        assert isinstance(result, Infinity)
        assert result.sign == -1

    def test_multiply_zero(self):
        """Test multiplying by zero (undefined)."""
        result = Infinity(1) * 0
        assert isinstance(result, Infinity)
        assert result.sign == 0

    def test_negate(self):
        """Test negation."""
        result = -Infinity(1)
        assert isinstance(result, Infinity)
        assert result.sign == -1

    def test_positive(self):
        """Test positive (no change)."""
        result = +Infinity(1)
        assert isinstance(result, Infinity)
        assert result.sign == 1


class TestInfinityTypePromotion:
    """Test type promotion behavior."""

    def test_promote_returns_self(self):
        """Test that Infinity doesn't promote."""
        inf = Infinity(1)
        assert inf.promote(None) is inf


class TestInfinityToPython:
    """Test conversion to Python native types."""

    def test_positive_to_python(self):
        """Test conversion to Python float."""
        val = Infinity(1).to_python()
        assert val == float("inf")

    def test_negative_to_python(self):
        """Test negative infinity to Python float."""
        val = Infinity(-1).to_python()
        assert val == float("-inf")

    def test_undefined_to_python(self):
        """Test undefined infinity to Python."""
        val = Infinity(0).to_python()
        assert str(val) == "nan"


class TestInfinityEdgeCases:
    """Test edge cases and special scenarios."""

    def test_division_not_implemented(self):
        """Test that division is properly handled."""
        inf = Infinity(1)
        result = inf / 2
        assert isinstance(result, Infinity)
        assert result.sign == 1

    def test_multiple_operations(self):
        """Test chaining multiple operations."""
        result = (Infinity(1) + 5) * 3
        assert isinstance(result, Infinity)
        assert result.sign == 1

    def test_context_preservation(self):
        """Test that context is preserved."""
        inf = Infinity(1)
        assert inf.context is not None

    def test_sign_bounds(self):
        """Test very large sign values normalize."""
        inf_pos = Infinity(1000)
        assert inf_pos.sign == 1

        inf_neg = Infinity(-1000)
        assert inf_neg.sign == -1
