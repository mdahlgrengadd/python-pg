"""Tests for Infinity MathValue."""

import math
import pytest

from pg.math.numeric import Infinity, Real


class TestInfinityConstruction:
    def test_default_is_positive_infinity(self):
        inf = Infinity()
        assert inf.sign == 1

    def test_negative_and_zero_signs(self):
        neg = Infinity(-5)
        zero = Infinity(0)
        assert neg.sign == -1
        assert zero.sign == 0

    def test_non_numeric_sign_raises(self):
        with pytest.raises(ValueError):
            Infinity("invalid")


class TestInfinityBehaviors:
    def test_compare_matching_signs(self):
        assert Infinity(1).compare(Infinity(1)) is True
        assert Infinity(1).compare(Infinity(-1)) is False

    def test_to_string_and_tex(self):
        assert Infinity(1).to_string() == "inf"
        assert Infinity(-1).to_tex() == r"-\infty"
        assert Infinity(0).to_string() == "NaN"

    def test_addition_with_infinity(self):
        assert (Infinity(1) + 5).sign == 1
        undefined = Infinity(1) + Infinity(-1)
        assert undefined.sign == 0

    def test_multiplication_by_real(self):
        result = Infinity(1) * Real(-2)
        assert isinstance(result, Infinity)
        assert result.sign == -1

    def test_division_by_infinity_returns_zero(self):
        zero = Infinity(1).__rtruediv__(2)
        assert isinstance(zero, Real)
        assert zero.value == 0.0

    def test_exponentiation_rules(self):
        assert (Infinity(1) ** 2).sign == 1
        assert isinstance((Infinity(1) ** -1), Real)
        assert math.isclose((Infinity(1) ** -1).value, 0.0)


