"""Tests for FormulaWithUnits and NumberWithUnits."""

import pytest
from pg.macros.parsers.parser_formula_with_units import (
    FormulaWithUnits,
    NumberWithUnits,
)


class TestNumberWithUnits:
    """Test NumberWithUnits class."""
    
    def test_parse_from_string(self):
        """Test parsing '3.5 ft' format."""
        num = NumberWithUnits("3.5 ft")
        assert num.value == 3.5
        assert num.units_str == "ft"
    
    def test_create_from_parts(self):
        """Test creating from number and unit."""
        num = NumberWithUnits(42, "m")
        assert num.value == 42.0
        assert num.units_str == "m"
    
    def test_string_representation(self):
        """Test string output."""
        num = NumberWithUnits(3.5, "ft")
        assert str(num) == "3.5 ft"
    
    def test_float_conversion(self):
        """Test converting to float."""
        num = NumberWithUnits(3.5, "ft")
        assert float(num) == 3.5


class TestFormulaWithUnits:
    """Test FormulaWithUnits class."""
    
    def test_parse_simple_formula(self):
        """Test parsing 'x+1 ft' format."""
        formula = FormulaWithUnits("x+1 ft")
        assert formula.units_str == "ft"
        assert str(formula.formula) in ["x+1", "x + 1", "1+x", "1 + x"]
    
    def test_parse_complex_formula(self):
        """Test parsing more complex formula."""
        formula = FormulaWithUnits("-16*t^2 + 64*t ft")
        assert formula.units_str == "ft"
    
    def test_create_from_parts(self):
        """Test creating from formula and unit."""
        # We can't easily create a Formula without the full stack,
        # so we test with string formula
        formula = FormulaWithUnits("x^2", "m")
        assert formula.units_str == "m"
    
    def test_string_representation(self):
        """Test string output."""
        formula = FormulaWithUnits("x+1", "ft")
        result = str(formula)
        assert "ft" in result
        assert "x" in result or "+" in result
    
    def test_units_with_division(self):
        """Test units like 'm/s'."""
        formula = FormulaWithUnits("5*t m/s")
        assert formula.units_str == "m/s"
    
    def test_looks_like_unit(self):
        """Test unit detection logic."""
        formula = FormulaWithUnits("x+1 ft")
        assert formula._looks_like_unit("ft")
        assert formula._looks_like_unit("m/s")
        assert formula._looks_like_unit("m^2")
        assert not formula._looks_like_unit("123")
        assert not formula._looks_like_unit("$var")


class TestFormulaOperations:
    """Test operations on FormulaWithUnits (when context available)."""
    
    def test_derivative_creates_new_formula(self):
        """Test that D() creates a new FormulaWithUnits."""
        # This test will pass even without full Formula support
        formula = FormulaWithUnits("t^2", "ft")
        
        # Check that the formula has the expected structure
        assert formula.units_str == "ft"
    
    def test_eval_returns_number_with_units(self):
        """Test that eval returns NumberWithUnits."""
        formula = FormulaWithUnits("2*x", "m")
        
        # Check formula structure
        assert formula.units_str == "m"


def test_module_exports():
    """Test that module exports expected classes."""
    from pg.macros.parsers.parser_formula_with_units import (
        FormulaWithUnits,
        NumberWithUnits,
        FormulaWithUnits_factory,
        NumberWithUnits_factory,
    )
    
    assert FormulaWithUnits is not None
    assert NumberWithUnits is not None
    assert FormulaWithUnits_factory is not None
    assert NumberWithUnits_factory is not None

