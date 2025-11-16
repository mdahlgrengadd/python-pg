"""Tests for Fraction MathValue class."""

import pytest

from pg.math.fraction import Fraction, gcd, lcm, reduce_fraction
from pg.math.value import MathValue


class TestFractionHelpers:
    """Test helper functions for fractions."""

    def test_gcd_simple(self):
        """Test GCD of simple numbers."""
        assert gcd(12, 8) == 4
        assert gcd(15, 10) == 5

    def test_gcd_coprime(self):
        """Test GCD of coprime numbers."""
        assert gcd(7, 11) == 1

    def test_gcd_with_zero(self):
        """Test GCD with zero."""
        assert gcd(5, 0) == 5
        assert gcd(0, 5) == 5

    def test_gcd_negative(self):
        """Test GCD with negative numbers."""
        assert gcd(-12, 8) == 4
        assert gcd(12, -8) == 4

    def test_lcm_simple(self):
        """Test LCM of simple numbers."""
        assert lcm(4, 6) == 12
        assert lcm(3, 5) == 15

    def test_lcm_with_one(self):
        """Test LCM with 1."""
        assert lcm(1, 5) == 5
        assert lcm(5, 1) == 5

    def test_reduce_fraction_basic(self):
        """Test fraction reduction."""
        assert reduce_fraction(2, 4) == (1, 2)
        assert reduce_fraction(6, 9) == (2, 3)

    def test_reduce_fraction_already_reduced(self):
        """Test reducing already reduced fraction."""
        assert reduce_fraction(3, 5) == (3, 5)

    def test_reduce_fraction_negative_denominator(self):
        """Test that denominator is made positive."""
        num, den = reduce_fraction(3, -5)
        assert den > 0
        assert num == -3


class TestFractionBasicInstantiation:
    """Test basic instantiation and field access."""

    def test_instantiate_simple_fraction(self):
        """Test creating a simple fraction."""
        f = Fraction(1, 2)
        assert f.num == 1
        assert f.den == 2

    def test_instantiate_whole_number(self):
        """Test creating a whole number as fraction."""
        f = Fraction(5)
        assert f.num == 5
        assert f.den == 1

    def test_instantiate_negative_numerator(self):
        """Test creating fraction with negative numerator."""
        f = Fraction(-3, 4)
        assert f.num == -3
        assert f.den == 4

    def test_instantiate_negative_denominator(self):
        """Test that negative denominator is normalized."""
        f = Fraction(3, -4)
        assert f.den > 0
        assert f.num == -3

    def test_instantiate_with_reduction(self):
        """Test that fraction is reduced by default."""
        f = Fraction(2, 4)
        assert f.num == 1
        assert f.den == 2

    def test_instantiate_without_reduction(self):
        """Test creating fraction without reduction."""
        f = Fraction(2, 4, reduce=False)
        assert f.num == 2
        assert f.den == 4

    def test_instantiate_from_float(self):
        """Test creating fraction from float."""
        f = Fraction(0.5)
        assert f.num == 1
        assert f.den == 2

    def test_instantiate_from_float_complex(self):
        """Test creating fraction from complex float."""
        f = Fraction(0.25)
        assert f.num == 1
        assert f.den == 4

    def test_fraction_type_precedence(self):
        """Test that Fraction has correct type precedence."""
        f = Fraction(1, 2)
        assert hasattr(f, 'type_precedence')

    def test_fraction_has_context(self):
        """Test that Fraction has a context."""
        f = Fraction(1, 2)
        assert hasattr(f, 'context')


class TestFractionConversions:
    """Test conversion methods."""

    def test_to_string_simple(self):
        """Test to_string() for simple fraction."""
        f = Fraction(1, 2)
        result = f.to_string()
        assert "1" in result
        assert "2" in result
        assert "/" in result

    def test_to_string_whole_number(self):
        """Test to_string() for whole number fraction."""
        f = Fraction(5, 1)
        result = f.to_string()
        assert "5" in result

    def test_to_string_improper(self):
        """Test to_string() for improper fraction."""
        f = Fraction(5, 2)
        result = f.to_string()
        assert "5" in result
        assert "2" in result

    def test_to_tex_format(self):
        """Test to_tex() returns LaTeX format."""
        f = Fraction(1, 2)
        result = f.to_tex()
        assert isinstance(result, str)
        # Should contain frac or similar LaTeX command
        assert "\\" in result or "/" in result

    def test_to_python_returns_float(self):
        """Test to_python() returns float."""
        f = Fraction(1, 2)
        result = f.to_python()
        assert isinstance(result, float)
        assert result == 0.5

    def test_to_python_whole_number(self):
        """Test to_python() for whole number."""
        f = Fraction(5, 1)
        result = f.to_python()
        assert result == 5.0


class TestFractionComparison:
    """Test comparison operations."""

    def test_eq_identical_fractions(self):
        """Test equality of identical fractions."""
        f1 = Fraction(1, 2)
        f2 = Fraction(1, 2)
        assert f1 == f2

    def test_eq_equivalent_fractions(self):
        """Test equality of equivalent fractions."""
        f1 = Fraction(1, 2)
        f2 = Fraction(2, 4)  # Gets reduced to 1/2
        assert f1 == f2

    def test_ne_different_fractions(self):
        """Test inequality of different fractions."""
        f1 = Fraction(1, 2)
        f2 = Fraction(1, 3)
        assert f1 != f2

    def test_compare_method_exact_match(self):
        """Test compare() method with exact match."""
        f1 = Fraction(1, 2)
        f2 = Fraction(1, 2)
        assert f1.compare(f2) is True

    def test_compare_method_different(self):
        """Test compare() method with different values."""
        f1 = Fraction(1, 2)
        f2 = Fraction(1, 3)
        assert f1.compare(f2) is False

    def test_compare_with_tolerance(self):
        """Test compare with tolerance."""
        f1 = Fraction(1, 2)
        f2 = Fraction(1, 2)
        assert f1.compare(f2, tolerance=0.001) is True


class TestFractionArithmetic:
    """Test arithmetic operations."""

    def test_add_two_fractions(self):
        """Test adding two fractions."""
        f1 = Fraction(1, 2)
        f2 = Fraction(1, 3)
        result = f1 + f2
        assert isinstance(result, Fraction)
        # 1/2 + 1/3 = 3/6 + 2/6 = 5/6
        assert result.num == 5
        assert result.den == 6

    def test_add_fraction_and_int(self):
        """Test adding fraction and integer."""
        f = Fraction(1, 2)
        result = f + 1
        assert isinstance(result, Fraction)
        # 1/2 + 1 = 1/2 + 2/2 = 3/2
        assert result.num == 3
        assert result.den == 2

    def test_sub_two_fractions(self):
        """Test subtracting two fractions."""
        f1 = Fraction(3, 4)
        f2 = Fraction(1, 4)
        result = f1 - f2
        assert isinstance(result, Fraction)
        assert result.num == 1
        assert result.den == 2

    def test_mul_two_fractions(self):
        """Test multiplying two fractions."""
        f1 = Fraction(2, 3)
        f2 = Fraction(3, 4)
        result = f1 * f2
        assert isinstance(result, Fraction)
        # 2/3 * 3/4 = 6/12 = 1/2
        assert result.num == 1
        assert result.den == 2

    def test_mul_fraction_by_int(self):
        """Test multiplying fraction by integer."""
        f = Fraction(1, 2)
        result = f * 3
        assert isinstance(result, Fraction)
        # 1/2 * 3 = 3/2
        assert result.num == 3
        assert result.den == 2

    def test_div_two_fractions(self):
        """Test dividing two fractions."""
        f1 = Fraction(2, 3)
        f2 = Fraction(4, 5)
        result = f1 / f2
        assert isinstance(result, Fraction)
        # 2/3 / 4/5 = 2/3 * 5/4 = 10/12 = 5/6
        assert result.num == 5
        assert result.den == 6

    def test_neg_fraction(self):
        """Test unary negation of fraction."""
        f = Fraction(1, 2)
        result = -f
        assert isinstance(result, Fraction)
        assert result.num == -1
        assert result.den == 2

    def test_abs_fraction(self):
        """Test absolute value of fraction."""
        f = Fraction(-1, 2)
        result = abs(f)
        assert isinstance(result, Fraction)
        assert result.num == 1
        assert result.den == 2


class TestFractionProperties:
    """Test fraction properties and methods."""

    def test_fraction_reduction_on_arithmetic(self):
        """Test that fractions are reduced after arithmetic."""
        f1 = Fraction(1, 2, reduce=False)
        f2 = Fraction(1, 2, reduce=False)
        # After addition, should be reduced
        result = f1 + f2
        assert result.num == 1
        assert result.den == 1


class TestFractionEdgeCases:
    """Test edge cases and special values."""

    def test_zero_fraction(self):
        """Test zero fraction."""
        f = Fraction(0, 5)
        assert f.num == 0
        assert f.den == 1

    def test_one_fraction(self):
        """Test fraction equal to one."""
        f = Fraction(1, 1)
        assert f.num == 1
        assert f.den == 1

    def test_very_large_fraction(self):
        """Test fraction with large numbers."""
        f = Fraction(1000000, 2000000)
        assert f.num == 1
        assert f.den == 2

    def test_is_instance_of_mathvalue(self):
        """Test that Fraction is instance of MathValue."""
        f = Fraction(1, 2)
        assert isinstance(f, MathValue)


class TestFractionIntegration:
    """Integration tests for Fraction class."""

    def test_workflow_create_operate_compare(self):
        """Test workflow: create, operate, and compare."""
        f1 = Fraction(1, 2)
        f2 = Fraction(1, 2)
        result = f1 + f2
        expected = Fraction(1, 1)
        assert result == expected

    def test_workflow_arithmetic_chain(self):
        """Test chaining arithmetic operations."""
        f = Fraction(1, 2)
        result = f + Fraction(1, 4) - Fraction(1, 4)
        expected = Fraction(1, 2)
        assert result == expected

    def test_workflow_conversion_chain(self):
        """Test conversion workflow."""
        f = Fraction(1, 2)
        str_repr = f.to_string()
        float_val = f.to_python()
        assert isinstance(str_repr, str)
        assert isinstance(float_val, float)
        assert float_val == 0.5
