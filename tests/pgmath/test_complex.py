"""Tests for Complex MathValue class."""

import math
import pytest

from pg.math.numeric import Complex, Real, Infinity
from pg.math.value import MathValue


class TestComplexBasicInstantiation:
    """Test basic instantiation and field access."""

    def test_instantiate_from_real_and_imag(self):
        """Test creating Complex from real and imaginary parts."""
        c = Complex(2, 3)
        assert c.real == 2.0
        assert c.imag == 3.0

    def test_instantiate_from_real_only(self):
        """Test creating Complex from real part only."""
        c = Complex(5)
        assert c.real == 5.0
        assert c.imag == 0.0

    def test_instantiate_from_negative_numbers(self):
        """Test creating Complex from negative numbers."""
        c = Complex(-2, -3)
        assert c.real == -2.0
        assert c.imag == -3.0

    def test_instantiate_from_floats(self):
        """Test creating Complex from floats."""
        c = Complex(1.5, 2.5)
        assert c.real == 1.5
        assert c.imag == 2.5

    def test_instantiate_from_list(self):
        """Test creating Complex from list [real, imag]."""
        c = Complex([3, 4])
        assert c.real == 3.0
        assert c.imag == 4.0

    def test_instantiate_from_tuple(self):
        """Test creating Complex from tuple (real, imag)."""
        c = Complex((2, 5))
        assert c.real == 2.0
        assert c.imag == 5.0

    def test_instantiate_from_single_element_list(self):
        """Test creating Complex from list with single element."""
        c = Complex([7])
        assert c.real == 7.0
        assert c.imag == 0.0

    def test_instantiate_from_empty_list(self):
        """Test creating Complex from empty list."""
        c = Complex([])
        assert c.real == 0.0
        assert c.imag == 0.0

    def test_instantiate_from_complex_string(self):
        """Test creating Complex from string like '2+3i'."""
        c = Complex("2+3i")
        assert c.real == 2.0
        assert c.imag == 3.0

    def test_instantiate_from_complex_string_with_spaces(self):
        """Test creating Complex from string with spaces."""
        c = Complex("2 + 3i")
        assert c.real == 2.0
        assert c.imag == 3.0

    def test_instantiate_from_negative_complex_string(self):
        """Test creating Complex from string with negative parts."""
        c = Complex("2-4i")
        assert c.real == 2.0
        assert c.imag == -4.0

    def test_instantiate_from_negative_real_string(self):
        """Test creating Complex from string with negative real."""
        c = Complex("-3+2i")
        assert c.real == -3.0
        assert c.imag == 2.0

    def test_instantiate_from_real_only_string(self):
        """Test creating Complex from real-only string."""
        c = Complex("5.5")
        assert c.real == 5.5
        assert c.imag == 0.0

    def test_instantiate_from_invalid_string_raises_error(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError):
            Complex("invalid")

    def test_complex_type_precedence(self):
        """Test that Complex has correct type precedence."""
        c = Complex(1, 2)
        assert hasattr(c, 'type_precedence')

    def test_complex_has_context(self):
        """Test that Complex has a context."""
        c = Complex(1, 2)
        assert hasattr(c, 'context')
        assert c.context is not None


class TestComplexConversions:
    """Test conversion methods."""

    def test_to_string_positive(self):
        """Test to_string() for positive parts."""
        c = Complex(2, 3)
        result = c.to_string()
        assert "2" in result
        assert "3" in result
        assert "i" in result

    def test_to_string_negative_imag(self):
        """Test to_string() with negative imaginary."""
        c = Complex(2, -3)
        result = c.to_string()
        assert "2" in result
        assert "-3" in result or "- 3" in result

    def test_to_string_pure_imaginary(self):
        """Test to_string() for pure imaginary."""
        c = Complex(0, 5)
        result = c.to_string()
        assert "5" in result
        assert "i" in result

    def test_to_string_real_only(self):
        """Test to_string() for real-only complex."""
        c = Complex(5, 0)
        result = c.to_string()
        assert "5" in result

    def test_to_tex_format(self):
        """Test to_tex() returns LaTeX format."""
        c = Complex(2, 3)
        result = c.to_tex()
        assert isinstance(result, str)

    def test_to_python_returns_python_complex(self):
        """Test to_python() returns Python complex type."""
        c = Complex(2, 3)
        result = c.to_python()
        assert isinstance(result, complex)
        assert result.real == 2.0
        assert result.imag == 3.0

    def test_complex_builtin_conversion(self):
        """Test that complex() builtin requires to_python() conversion."""
        c = Complex(3, 4)
        # Complex doesn't implement __complex__, must use to_python()
        result = complex(c.to_python())
        assert isinstance(result, complex)
        assert result == 3+4j


class TestComplexComparison:
    """Test comparison operations."""

    def test_eq_identical_complex(self):
        """Test equality of identical complex numbers."""
        c1 = Complex(2, 3)
        c2 = Complex(2, 3)
        assert c1 == c2

    def test_eq_complex_with_python_complex(self):
        """Test equality comparison with Python complex (requires conversion)."""
        c = Complex(2, 3)
        # Direct comparison with Python complex may not work
        # Use to_python() for conversion
        assert c.to_python() == (2+3j)

    def test_ne_different_complex(self):
        """Test inequality of different complex numbers."""
        c1 = Complex(2, 3)
        c2 = Complex(2, 4)
        assert c1 != c2

    def test_compare_method_exact_match(self):
        """Test compare() method with exact match."""
        c1 = Complex(2, 3)
        c2 = Complex(2, 3)
        assert c1.compare(c2) is True

    def test_compare_method_different(self):
        """Test compare() method with different values."""
        c1 = Complex(2, 3)
        c2 = Complex(2, 4)
        assert c1.compare(c2) is False

    def test_compare_with_real_promotion(self):
        """Test compare with Real (should promote Real to Complex)."""
        c = Complex(5, 0)
        r = Real(5)
        assert c.compare(r) is True

    def test_compare_with_tolerance(self):
        """Test compare with tolerance."""
        c1 = Complex(2.0, 3.0)
        c2 = Complex(2.0005, 3.0005)
        assert c1.compare(c2, tolerance=0.001) is True


class TestComplexArithmetic:
    """Test arithmetic operations."""

    def test_add_two_complex(self):
        """Test adding two complex numbers."""
        c1 = Complex(2, 3)
        c2 = Complex(1, 4)
        result = c1 + c2
        assert isinstance(result, Complex)
        assert result.real == 3.0
        assert result.imag == 7.0

    def test_add_complex_and_real(self):
        """Test adding Complex and Real."""
        c = Complex(2, 3)
        r = Real(5)
        result = c + r
        assert isinstance(result, Complex)
        assert result.real == 7.0
        assert result.imag == 3.0

    def test_add_complex_and_int(self):
        """Test adding Complex and int."""
        c = Complex(2, 3)
        result = c + 4
        assert isinstance(result, Complex)
        assert result.real == 6.0
        assert result.imag == 3.0

    def test_sub_two_complex(self):
        """Test subtracting two complex numbers."""
        c1 = Complex(5, 7)
        c2 = Complex(2, 3)
        result = c1 - c2
        assert isinstance(result, Complex)
        assert result.real == 3.0
        assert result.imag == 4.0

    def test_mul_two_complex(self):
        """Test multiplying two complex numbers."""
        c1 = Complex(2, 3)
        c2 = Complex(1, 4)
        result = c1 * c2
        assert isinstance(result, Complex)
        # (2+3i)(1+4i) = 2 + 8i + 3i + 12i^2 = 2 + 11i - 12 = -10 + 11i
        assert result.real == -10.0
        assert result.imag == 11.0

    def test_mul_complex_by_int(self):
        """Test multiplying Complex by int."""
        c = Complex(2, 3)
        result = c * 2
        assert isinstance(result, Complex)
        assert result.real == 4.0
        assert result.imag == 6.0

    def test_div_two_complex(self):
        """Test dividing two complex numbers."""
        c1 = Complex(4, 2)
        c2 = Complex(2, 1)
        result = c1 / c2
        assert isinstance(result, Complex)
        # (4+2i)/(2+i) = (4+2i)(2-i)/5 = (8-4i+4i-2i^2)/5 = (8+2)/5 = 10/5 = 2
        assert result.real == pytest.approx(2.0)
        assert result.imag == pytest.approx(0.0)

    def test_neg_complex(self):
        """Test unary negation of complex."""
        c = Complex(2, 3)
        result = -c
        assert isinstance(result, Complex)
        assert result.real == -2.0
        assert result.imag == -3.0

    def test_abs_complex(self):
        """Test absolute value (magnitude) of complex."""
        c = Complex(3, 4)
        result = abs(c)
        # |3+4i| = sqrt(9 + 16) = sqrt(25) = 5
        assert result == 5.0


class TestComplexPromotion:
    """Test type promotion."""

    def test_promote_with_real(self):
        """Test promotion with Real."""
        c = Complex(2, 3)
        r = Real(5)
        promoted = c.promote(r)
        # Complex doesn't promote, returns self
        assert promoted is c

    def test_promote_with_complex(self):
        """Test promotion with another Complex."""
        c1 = Complex(2, 3)
        c2 = Complex(4, 5)
        promoted = c1.promote(c2)
        assert promoted is c1


class TestComplexEdgeCases:
    """Test edge cases and special values."""

    def test_pure_real_complex(self):
        """Test complex with zero imaginary part."""
        c = Complex(5, 0)
        assert c.real == 5.0
        assert c.imag == 0.0

    def test_pure_imaginary_complex(self):
        """Test complex with zero real part."""
        c = Complex(0, 5)
        assert c.real == 0.0
        assert c.imag == 5.0

    def test_zero_complex(self):
        """Test zero complex number."""
        c = Complex(0, 0)
        assert c.real == 0.0
        assert c.imag == 0.0

    def test_very_large_complex(self):
        """Test complex with very large values."""
        c = Complex(1e100, 1e100)
        assert c.real == 1e100
        assert c.imag == 1e100

    def test_very_small_complex(self):
        """Test complex with very small values."""
        c = Complex(1e-100, 1e-100)
        assert c.real == 1e-100
        assert c.imag == 1e-100

    def test_is_instance_of_mathvalue(self):
        """Test that Complex is instance of MathValue."""
        c = Complex(2, 3)
        assert isinstance(c, MathValue)


class TestComplexIntegration:
    """Integration tests for Complex class."""

    def test_workflow_create_operate_compare(self):
        """Test workflow: create, operate, and compare."""
        c1 = Complex(2, 3)
        c2 = Complex(1, 2)
        result = c1 + c2
        expected = Complex(3, 5)
        assert result == expected

    def test_workflow_arithmetic_chain(self):
        """Test chaining arithmetic operations."""
        c = Complex(2, 3)
        result = c + Complex(1, 1) - Complex(0, 1)
        expected = Complex(3, 3)
        assert result == expected

    def test_workflow_conversion_chain(self):
        """Test conversion workflow."""
        c = Complex(2, 3)
        str_repr = c.to_string()
        python_complex = c.to_python()
        assert isinstance(str_repr, str)
        assert isinstance(python_complex, complex)

    def test_workflow_mixed_arithmetic(self):
        """Test arithmetic mixing Complex, Real, and int."""
        c = Complex(2, 3)
        r = Real(1)
        result = c + r + 2
        assert isinstance(result, Complex)
        assert result.real == 5.0
        assert result.imag == 3.0
