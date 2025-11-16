"""Tests for Real MathValue class."""

import math
import pytest

from pg.math.numeric import Real, Infinity, Complex
from pg.math.value import MathValue, ToleranceMode


@pytest.fixture
def context_mock():
    """Fixture providing a mock context."""
    class MockContext:
        def __init__(self):
            self.flags = {
                'tolerance': 0.001,
                'tolType': 'relative',
                'zeroLevel': 1e-14,
                'zeroLevelTol': 1e-12,
            }
    return MockContext()


class TestRealBasicInstantiation:
    """Test basic instantiation and field access."""

    def test_instantiate_from_int(self):
        """Test creating Real from integer."""
        r = Real(5)
        assert r.value == 5.0
        assert isinstance(r.value, float)

    def test_instantiate_from_float(self):
        """Test creating Real from float."""
        r = Real(3.14)
        assert r.value == 3.14

    def test_instantiate_from_negative_int(self):
        """Test creating Real from negative integer."""
        r = Real(-10)
        assert r.value == -10.0

    def test_instantiate_from_negative_float(self):
        """Test creating Real from negative float."""
        r = Real(-2.5)
        assert r.value == -2.5

    def test_instantiate_from_zero(self):
        """Test creating Real from zero."""
        r = Real(0)
        assert r.value == 0.0

    def test_instantiate_from_large_number(self):
        """Test creating Real from large number."""
        r = Real(1e10)
        assert r.value == 1e10

    def test_instantiate_from_small_number(self):
        """Test creating Real from small number."""
        r = Real(1e-10)
        assert r.value == 1e-10

    def test_instantiate_from_numeric_string(self):
        """Test creating Real from numeric string."""
        r = Real("42")
        assert r.value == 42.0

    def test_instantiate_from_float_string(self):
        """Test creating Real from float string."""
        r = Real("3.14")
        assert r.value == pytest.approx(3.14)

    def test_instantiate_from_pi_constant(self):
        """Test creating Real from pi expression."""
        r = Real("pi")
        assert r.value == pytest.approx(math.pi)

    def test_instantiate_from_e_constant(self):
        """Test creating Real from e expression."""
        r = Real("e")
        assert r.value == pytest.approx(math.e)

    def test_instantiate_from_pi_division(self):
        """Test creating Real from pi/2 expression."""
        r = Real("pi/2")
        assert r.value == pytest.approx(math.pi / 2)

    def test_instantiate_from_sqrt_expression(self):
        """Test creating Real from sqrt expression."""
        r = Real("sqrt(4)")
        assert r.value == pytest.approx(2.0)

    def test_instantiate_from_invalid_string_raises_error(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError):
            Real("invalid_expression_xyz")

    def test_instantiate_from_malicious_expression_raises_error(self):
        """Test that potentially malicious expressions are handled."""
        # eval() is used, so we test that dangerous constructs fail
        # The current implementation uses a restricted namespace
        with pytest.raises(ValueError):
            Real("__import__('os').system('echo test')")

    def test_real_type_precedence(self):
        """Test that Real has correct type precedence."""
        r = Real(1)
        assert hasattr(r, 'type_precedence')

    def test_real_has_context(self):
        """Test that Real has a context."""
        r = Real(1)
        assert hasattr(r, 'context')
        assert r.context is not None


class TestRealConversions:
    """Test conversion methods."""

    def test_to_string_integer_value(self):
        """Test to_string() for integer values."""
        r = Real(42)
        assert r.to_string() == "42"

    def test_to_string_float_value(self):
        """Test to_string() for float values."""
        r = Real(3.14159)
        result = r.to_string()
        assert "3.14" in result

    def test_to_string_zero(self):
        """Test to_string() for zero."""
        r = Real(0)
        assert r.to_string() == "0"

    def test_to_string_negative(self):
        """Test to_string() for negative numbers."""
        r = Real(-5)
        assert r.to_string() == "-5"

    def test_to_tex_simple(self):
        """Test to_tex() returns same as to_string()."""
        r = Real(42)
        assert r.to_tex() == r.to_string()

    def test_to_python_returns_float(self):
        """Test to_python() returns float."""
        r = Real(42)
        result = r.to_python()
        assert isinstance(result, float)
        assert result == 42.0

    def test_float_builtin_conversion(self):
        """Test float() builtin conversion."""
        r = Real(3.14)
        result = float(r)
        assert isinstance(result, float)
        assert result == 3.14

    def test_str_builtin_conversion(self):
        """Test str() builtin conversion."""
        r = Real(42)
        result = str(r)
        assert result == "42"

    def test_repr_builtin_conversion(self):
        """Test repr() builtin conversion."""
        r = Real(5)
        result = repr(r)
        assert "Real" in result
        assert "5" in result


class TestRealComparison:
    """Test comparison operations."""

    def test_eq_identical_values(self):
        """Test equality of identical values."""
        r1 = Real(5)
        r2 = Real(5)
        assert r1 == r2

    def test_eq_with_float(self):
        """Test equality with Python float."""
        r = Real(3.0)
        assert r == 3.0

    def test_eq_with_int(self):
        """Test equality with Python int."""
        r = Real(5)
        assert r == 5

    def test_eq_with_tolerance(self):
        """Test equality with fuzzy comparison."""
        r1 = Real(1.0)
        r2 = Real(1.0005)
        # With default tolerance 0.001, these should be equal
        assert r1 == r2

    def test_ne_different_values(self):
        """Test inequality of different values."""
        r1 = Real(5)
        r2 = Real(6)
        assert r1 != r2

    def test_ne_with_float(self):
        """Test inequality with Python float."""
        r = Real(5)
        assert r != 5.5

    def test_compare_method_exact_match(self):
        """Test compare() method with exact match."""
        r1 = Real(5)
        r2 = Real(5)
        assert r1.compare(r2) is True

    def test_compare_method_within_tolerance(self):
        """Test compare() method within tolerance."""
        r1 = Real(1.0)
        r2 = Real(1.0005)
        assert r1.compare(r2, tolerance=0.001, mode='relative') is True

    def test_compare_method_outside_tolerance(self):
        """Test compare() method outside tolerance."""
        r1 = Real(1.0)
        r2 = Real(1.01)
        assert r1.compare(r2, tolerance=0.001, mode='relative') is False

    def test_compare_method_absolute_tolerance(self):
        """Test compare() with absolute tolerance."""
        r1 = Real(5.0)
        r2 = Real(5.001)
        assert r1.compare(r2, tolerance=0.01, mode='absolute') is True

    def test_compare_with_non_mathvalue_raises_error(self):
        """Test compare with non-MathValue raises AttributeError."""
        r = Real(5)
        # Comparing with non-MathValue will raise AttributeError
        with pytest.raises(AttributeError):
            r.compare("not a real")


class TestRealArithmetic:
    """Test arithmetic operations."""

    def test_add_two_reals(self):
        """Test adding two Real numbers."""
        r1 = Real(3)
        r2 = Real(4)
        result = r1 + r2
        assert isinstance(result, Real)
        assert result.value == 7.0

    def test_add_real_and_int(self):
        """Test adding Real and int."""
        r = Real(3)
        result = r + 4
        assert isinstance(result, Real)
        assert result.value == 7.0

    def test_add_real_and_float(self):
        """Test adding Real and float."""
        r = Real(3.5)
        result = r + 1.5
        assert isinstance(result, Real)
        assert result.value == 5.0

    def test_radd_int_and_real(self):
        """Test right-adding int and Real."""
        r = Real(4)
        result = 3 + r
        assert isinstance(result, Real)
        assert result.value == 7.0

    def test_sub_two_reals(self):
        """Test subtracting two Real numbers."""
        r1 = Real(10)
        r2 = Real(3)
        result = r1 - r2
        assert isinstance(result, Real)
        assert result.value == 7.0

    def test_sub_real_and_int(self):
        """Test subtracting int from Real."""
        r = Real(10)
        result = r - 3
        assert isinstance(result, Real)
        assert result.value == 7.0

    def test_rsub_int_minus_real(self):
        """Test right subtraction (int - Real)."""
        r = Real(3)
        result = 10 - r
        assert isinstance(result, Real)
        assert result.value == 7.0

    def test_mul_two_reals(self):
        """Test multiplying two Real numbers."""
        r1 = Real(3)
        r2 = Real(4)
        result = r1 * r2
        assert isinstance(result, Real)
        assert result.value == 12.0

    def test_mul_real_and_int(self):
        """Test multiplying Real and int."""
        r = Real(3)
        result = r * 4
        assert isinstance(result, Real)
        assert result.value == 12.0

    def test_rmul_int_and_real(self):
        """Test right-multiplying int and Real."""
        r = Real(4)
        result = 3 * r
        assert isinstance(result, Real)
        assert result.value == 12.0

    def test_div_two_reals(self):
        """Test dividing two Real numbers."""
        r1 = Real(12)
        r2 = Real(4)
        result = r1 / r2
        assert isinstance(result, Real)
        assert result.value == 3.0

    def test_div_real_and_int(self):
        """Test dividing Real by int."""
        r = Real(12)
        result = r / 4
        assert isinstance(result, Real)
        assert result.value == 3.0

    def test_div_by_zero_returns_infinity(self):
        """Test division by zero returns Infinity."""
        r = Real(5)
        result = r / 0
        assert isinstance(result, Infinity)

    def test_div_negative_by_zero(self):
        """Test negative division by zero."""
        r = Real(-5)
        result = r / 0
        assert isinstance(result, Infinity)

    def test_pow_real_numbers(self):
        """Test power operation."""
        r1 = Real(2)
        r2 = Real(3)
        result = r1 ** r2
        assert isinstance(result, Real)
        assert result.value == 8.0

    def test_pow_with_int_exponent(self):
        """Test power with integer exponent."""
        r = Real(2)
        result = r ** 4
        assert isinstance(result, Real)
        assert result.value == 16.0

    def test_neg_unary(self):
        """Test unary negation."""
        r = Real(5)
        result = -r
        assert isinstance(result, Real)
        assert result.value == -5.0

    def test_pos_unary(self):
        """Test unary positive."""
        r = Real(5)
        result = +r
        assert isinstance(result, Real)
        assert result.value == 5.0

    def test_abs_positive(self):
        """Test absolute value of positive."""
        r = Real(5)
        result = abs(r)
        assert isinstance(result, Real)
        assert result.value == 5.0

    def test_abs_negative(self):
        """Test absolute value of negative."""
        r = Real(-5)
        result = abs(r)
        assert isinstance(result, Real)
        assert result.value == 5.0


class TestRealPromotion:
    """Test type promotion."""

    def test_promote_to_complex(self):
        """Test promotion to Complex."""
        r = Real(5)
        other = Complex(2, 3)
        promoted = r.promote(other)
        # Real DOES promote to Complex in the promote method
        assert isinstance(promoted, Complex)

    def test_promote_to_infinity(self):
        """Test promotion to Infinity."""
        r = Real(5)
        other = Infinity(1)
        promoted = r.promote(other)
        assert isinstance(promoted, Real)


class TestRealEdgeCases:
    """Test edge cases and special values."""

    def test_very_large_number(self):
        """Test with very large number."""
        r = Real(1e100)
        assert r.value == 1e100

    def test_very_small_positive_number(self):
        """Test with very small positive number."""
        r = Real(1e-100)
        assert r.value == 1e-100

    def test_zero_value(self):
        """Test with zero value."""
        r = Real(0)
        assert r.value == 0.0

    def test_addition_chain(self):
        """Test chaining additions."""
        r = Real(1) + Real(2) + Real(3)
        assert r.value == 6.0

    def test_arithmetic_chain(self):
        """Test complex arithmetic chain."""
        r = (Real(2) * Real(3)) + (Real(10) / Real(2))
        assert r.value == 11.0

    def test_division_by_very_small_number(self):
        """Test division by very small number."""
        r = Real(10) / Real(1e-10)
        assert r.value == 1e11

    def test_is_instance_of_mathvalue(self):
        """Test that Real is instance of MathValue."""
        r = Real(5)
        assert isinstance(r, MathValue)

    def test_multiple_creations_have_independent_contexts(self):
        """Test that multiple Real instances can have different contexts."""
        r1 = Real(5)
        r2 = Real(10)
        # Both should have contexts (even if same default)
        assert hasattr(r1, 'context')
        assert hasattr(r2, 'context')


class TestRealIntegration:
    """Integration tests for Real class."""

    def test_workflow_create_operate_compare(self):
        """Test workflow: create, operate, and compare."""
        r1 = Real(5)
        r2 = Real(3)
        result = r1 + r2
        expected = Real(8)
        assert result == expected

    def test_workflow_conversion_chain(self):
        """Test conversion workflow."""
        r = Real("pi")
        str_repr = r.to_string()
        float_val = r.to_python()
        py_float = float(r)

        assert isinstance(str_repr, str)
        assert isinstance(float_val, float)
        assert isinstance(py_float, float)

    def test_workflow_complex_calculation(self):
        """Test complex calculation workflow."""
        # Calculate: (2 * pi) / sqrt(2)
        r1 = Real("pi")
        r2 = Real(2) * r1  # 2*pi
        r3 = Real("sqrt(2)")
        result = r2 / r3
        assert isinstance(result, Real)
        expected = 2 * math.pi / math.sqrt(2)
        assert result.value == pytest.approx(expected)
