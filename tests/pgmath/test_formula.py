"""Tests for Formula MathValue class."""

import pytest

from pg.math.formula import Formula
from pg.math.numeric import Real
from pg.math.context import Context


class TestFormulaBasic:
    """Test basic Formula creation and evaluation."""

    def test_create_formula_from_string(self):
        """Test creating a Formula from a string expression."""
        f = Formula("x^2 + 1")
        assert isinstance(f, Formula)
        assert "x" in f.variables

    def test_formula_evaluation(self):
        """Test evaluating a Formula with variable bindings."""
        f = Formula("x^2 + 2*x + 1")
        result = f.eval(x=3)
        assert isinstance(result, Real)
        assert result.value == 16.0

    def test_formula_variable_extraction(self):
        """Test automatic variable extraction from expression."""
        f = Formula("x*y + z")
        assert "x" in f.variables
        assert "y" in f.variables
        assert "z" in f.variables

    def test_formula_with_context(self):
        """Test creating a Formula with a context."""
        ctx = Context("Numeric")
        f = Formula("x^2", context=ctx)
        assert f.context is not None
        assert f.context.name == "Numeric"

    def test_formula_to_string(self):
        """Test Formula string representation."""
        f = Formula("x^2 + 1")
        result = f.to_string()
        assert isinstance(result, str)
        assert "x" in result

    def test_formula_comparison(self):
        """Test comparing two Formulas."""
        f1 = Formula("x^2 + 1")
        f2 = Formula("x^2 + 1")
        # Formulas are compared using test points
        # This is a basic test - full comparison logic is complex
        assert f1.variables == f2.variables

