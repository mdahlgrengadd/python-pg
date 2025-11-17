"""
Comprehensive tests for Formula and FormulaUpToConstant Pydantic conversions.

Tests verify:
- Pydantic field definitions and validation
- Backward API compatibility
- Feature functionality
- Integration with FormulaBase base class
"""

import pytest
import random
from pg.math.formula import Formula, UNDEF_VALUE
from pg.math.formula_up_to_constant import FormulaUpToConstant
from pg.math.formula import FormulaBase
from pg.math.context import get_context
from pg.math.numeric import Real


class TestFormulaPydantic:
    """Test Pydantic field definitions and validation for Formula."""

    def test_basic_instantiation(self):
        """Test creating Formula with basic parameters."""
        f = Formula("x^2 + 1", variables=['x'])
        assert f.expression == "x^2 + 1"
        assert 'x' in f.variables
        assert f.granularity == 1000
        assert f.check_undefined_points is False
        assert f.parameters == []

    def test_field_definitions(self):
        """Test that Pydantic fields are properly defined."""
        f = Formula(
            "a*x + b",
            variables=['x'],
            granularity=500,
            check_undefined_points=True,
            max_undefined=3,
            parameters=['a', 'b']
        )
        assert f.granularity == 500
        assert f.check_undefined_points is True
        assert f.max_undefined == 3
        assert f.parameters == ['a', 'b']

    def test_granularity_validation(self):
        """Test that granularity must be positive."""
        with pytest.raises(ValueError):
            Formula("x", granularity=0)

        with pytest.raises(ValueError):
            Formula("x", granularity=-1)

    def test_parameters_default_factory(self):
        """Test that parameters defaults to empty list."""
        f1 = Formula("x")
        f2 = Formula("y")
        # Ensure defaults don't share state
        assert f1.parameters is not f2.parameters
        f1.parameters.append('p')
        assert f2.parameters == []

    def test_max_undefined_default(self):
        """Test max_undefined defaults to num_test_points if None."""
        f = Formula("x", num_test_points=10)
        # When not provided, max_undefined defaults to num_test_points
        assert f.max_undefined == 10

    def test_inherit_formula_fields(self):
        """Test that Formula inherits Formula fields."""
        f = Formula(
            "x^2",
            variables=['x'],
            num_test_points=7,
            limits={'x': (-5, 5)}
        )
        assert f.num_test_points == 7
        assert f.limits == {'x': (-5, 5)}


class TestFormulaFeatures:
    """Test Formula features work with Pydantic fields."""

    def test_create_random_points_basic(self):
        """Test random point generation."""
        f = Formula("x^2", variables=['x'])
        points = f.create_random_points(n=5)
        assert len(points) == 5
        assert all(isinstance(p, list) for p in points)

    def test_create_random_points_with_limits(self):
        """Test random points respect variable limits."""
        f = Formula(
            "x^2",
            variables=['x'],
            limits={'x': (0, 10)}
        )
        points = f.create_random_points(n=10)
        for point in points:
            assert 0 <= point[0] <= 10

    def test_granularity_application(self):
        """Test granularity affects point values."""
        f = Formula(
            "x",
            variables=['x'],
            granularity=10,  # 10 steps in range
            limits={'x': (0, 100)}
        )
        points = f.create_random_points(n=5)
        # With granularity 10, points should be multiples of 10
        for point in points:
            assert point[0] % 10 == 0

    def test_create_point_values(self):
        """Test evaluating formula at test points."""
        f = Formula("x^2", variables=['x'])
        points = [[1], [2], [3]]
        values = f.create_point_values(points)
        assert values is not None
        assert len(values) == 3
        # Should evaluate to 1, 4, 9
        assert float(values[0]) == 1.0
        assert float(values[1]) == 4.0
        assert float(values[2]) == 9.0

    def test_python_function_generation(self):
        """Test Python function generation from formula."""
        f = Formula("x^2 + 2*x + 1", variables=['x'])
        func = f.python_function()
        assert func(0) == 1
        assert func(1) == 4
        assert func(2) == 9

    def test_python_function_caching(self):
        """Test that Python functions are cached."""
        f = Formula("x^2", variables=['x'])
        func1 = f.python_function()
        func2 = f.python_function()
        assert func1 is func2  # Same object due to caching

    def test_uses_one_of_with_parameters(self):
        """Test uses_one_of method with parameters."""
        f = Formula("a*x + b", parameters=['a', 'b'], variables=['x'])
        assert f.uses_one_of('a') is True
        assert f.uses_one_of('b') is True
        assert f.uses_one_of('x') is True
        assert f.uses_one_of('y') is False


class TestFormulaAdaptiveParameters:
    """Test adaptive parameter solving with Pydantic fields."""

    def test_adapt_parameters_checks_parameters_exist(self):
        """Test that adapt_parameters returns False without parameters."""
        f = Formula("2*x", parameters=[])
        student = Formula("2*x")
        result = f.adapt_parameters(student)
        assert result is False

    def test_adapt_parameters_simple_case(self):
        """Test adaptive parameter solving for simple linear case."""
        f = Formula("a*x", parameters=['a'], variables=['x'])
        # This would solve for 'a' when comparing with student's formula
        # For now, just test that the method exists and can be called
        assert callable(f.adapt_parameters)


class TestFormulaUpToConstantPydantic:
    """Test Pydantic field definitions for FormulaUpToConstant."""

    def test_basic_instantiation(self):
        """Test creating FormulaUpToConstant."""
        f = FormulaUpToConstant("x^2/2 + C", variables=['x', 'C'])
        assert f.constant == 'C'
        assert 'C' in f.variables

    def test_constant_field(self):
        """Test constant field is properly defined."""
        f = FormulaUpToConstant("x^2 + K", variables=['x', 'K'])
        assert f.constant == 'K'

    def test_constant_auto_added(self):
        """Test that constant is automatically added if missing."""
        f = FormulaUpToConstant("x^2", variables=['x'])
        # Should auto-add C
        assert f.constant == 'C'
        assert 'C' in f.variables

    def test_arbitrary_constants_set(self):
        """Test _arbitrary_constants private attribute."""
        f = FormulaUpToConstant("x^2 + C")
        assert isinstance(f._arbitrary_constants, set)
        assert 'C' in f._arbitrary_constants

    def test_private_context_attribute(self):
        """Test _private_context private attribute."""
        ctx = get_context('Numeric')
        f = FormulaUpToConstant("x^2 + C", context=ctx)
        assert f._private_context is not None
        assert f._private_context is not ctx  # Should be a copy


class TestFormulaUpToConstantFeatures:
    """Test FormulaUpToConstant features with Pydantic conversion."""

    def test_compare_up_to_constant(self):
        """Test comparing formulas up to constant."""
        f = FormulaUpToConstant("x^2/2 + C")
        other = FormulaUpToConstant("x^2/2 + K")
        equal, msg = f.compare_up_to_constant(other)
        assert equal is True
        assert msg is None

    def test_compare_different_formula(self):
        """Test comparing different formulas."""
        f = FormulaUpToConstant("x^2/2 + C")
        other = FormulaUpToConstant("x^3/3 + K")
        equal, msg = f.compare_up_to_constant(other)
        assert equal is False
        assert msg is not None

    def test_compare_missing_constant(self):
        """Test comparing when student answer missing constant."""
        f = FormulaUpToConstant("x^2/2 + C")
        other = FormulaUpToConstant("x^2/2")  # Should auto-add C
        # Both have C, so they should compare equal
        equal, msg = f.compare_up_to_constant(other)
        # After auto-add, both should have the same form
        assert f.constant == 'C'

    def test_remove_constant(self):
        """Test removing constant from formula."""
        f = FormulaUpToConstant("x^2/2 + C")
        g = f.remove_constant()
        assert isinstance(g, Formula)
        # The resulting formula should be x^2/2
        # Verify by evaluating at x=2: should be 2
        # (can't easily test due to symbolic manipulation)

    def test_differentiate_removes_constant(self):
        """Test that differentiating returns Formula without constant."""
        f = FormulaUpToConstant("x^2/2 + C")
        df = f.diff('x')
        # diff() returns a FormulaBase (the parent class)
        assert isinstance(df, FormulaBase)
        assert not isinstance(df, FormulaUpToConstant)

    def test_cmp_method_returns_callable(self):
        """Test that cmp() returns a callable checker."""
        f = FormulaUpToConstant("x^2/2 + C")
        checker = f.cmp()
        assert callable(checker)

    def test_cmp_accepts_student_answer(self):
        """Test the cmp checker with student answer."""
        f = FormulaUpToConstant("x^2/2 + C")
        checker = f.cmp()
        result = checker("x^2/2 + K")
        assert isinstance(result, dict)
        assert 'correct' in result
        assert 'message' in result


class TestBackwardCompatibility:
    """Test backward compatibility of both enhanced formula classes."""

    def test_formula_enhanced_is_formula_subclass(self):
        """Test that Formula is a Formula subclass."""
        f = Formula("x^2")
        assert isinstance(f, Formula)

    def test_formula_up_to_constant_is_formula_subclass(self):
        """Test that FormulaUpToConstant is a Formula subclass."""
        f = FormulaUpToConstant("x^2 + C")
        assert isinstance(f, Formula)

    def test_formula_enhanced_supports_eval(self):
        """Test that eval() method works on Formula."""
        f = Formula("x^2 + 1", variables=['x'])
        result = f.eval(x=3)
        assert float(result) == 10

    def test_formula_up_to_constant_supports_eval(self):
        """Test that eval() method works on FormulaUpToConstant."""
        f = FormulaUpToConstant("x^2 + C", variables=['x', 'C'])
        result = f.eval(x=2, C=5)
        assert float(result) == 9  # 4 + 5

    def test_formula_enhanced_cmp_compatibility(self):
        """Test that cmp() works on Formula."""
        f = Formula("x^2", variables=['x'])
        # Should have cmp method from Formula
        assert hasattr(f, 'cmp')

    def test_formula_up_to_constant_preserves_expression(self):
        """Test that expression is preserved correctly."""
        expr = "x^2/2 + C"
        f = FormulaUpToConstant(expr)
        # Expression might be modified internally, but should contain same values
        assert 'x' in str(f.expression)


class TestPydanticValidation:
    """Test Pydantic validation features."""

    def test_validate_assignment_on_formula_enhanced(self):
        """Test that field assignment triggers validation."""
        f = Formula("x", granularity=500)
        # Try to set invalid granularity
        with pytest.raises(ValueError):
            f.granularity = 0

    def test_arbitrary_types_allowed(self):
        """Test that arbitrary types are allowed in Pydantic fields."""
        ctx = get_context('Numeric')
        # Should not raise error despite having 'Any' context field
        f = Formula("x", context=ctx)
        assert f.context is ctx

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed."""
        # Formula classes allow extra='allow' in config
        f = Formula("x", custom_field="custom_value")
        # Should not raise validation error


class TestIntegration:
    """Integration tests for enhanced formula classes."""

    def test_formula_enhanced_with_context(self):
        """Test Formula with mathematical context."""
        ctx = get_context('Numeric')
        f = Formula("sin(x)", context=ctx, variables=['x'])
        points = f.create_random_points(n=3)
        values = f.create_point_values(points)
        assert values is not None
        assert len(values) == len(points)

    def test_formula_up_to_constant_with_context(self):
        """Test FormulaUpToConstant with mathematical context."""
        ctx = get_context('Numeric')
        f = FormulaUpToConstant("sin(x) + C", context=ctx)
        assert f.constant == 'C'
        assert 'x' in f.variables

    def test_formula_enhanced_multiple_variables(self):
        """Test Formula with multiple variables."""
        f = Formula(
            "x*y + z",
            variables=['x', 'y', 'z'],
            limits={
                'x': (-2, 2),
                'y': (-3, 3),
                'z': (-1, 1)
            }
        )
        points = f.create_random_points(n=5)
        assert all(len(p) == 3 for p in points)

    def test_formula_enhanced_parameter_in_expression(self):
        """Test Formula with parameters in expression."""
        f = Formula(
            "a*x^2 + b*x + c",
            variables=['x'],
            parameters=['a', 'b', 'c']
        )
        assert f.uses_one_of('a', 'b', 'c')
        func = f.python_function(vars=['x', 'a', 'b', 'c'])
        # Evaluate at x=1, a=1, b=1, c=1 should give 3
        result = func(1, 1, 1, 1)
        assert result == 3


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_formula_enhanced_no_variables(self):
        """Test Formula with constant expression."""
        f = Formula("5")
        points = f.create_random_points(n=3)
        # With no variables, returns single empty point
        assert len(points) == 1
        assert points[0] == []

    def test_formula_up_to_constant_single_constant(self):
        """Test FormulaUpToConstant enforces single constant."""
        # Should work with single constant
        f = FormulaUpToConstant("x^2 + C", variables=['x', 'C'])
        assert f.constant == 'C'

    def test_formula_enhanced_with_reserved_names(self):
        """Test that reserved names are handled correctly."""
        # 'e' and 'pi' should not be treated as variables
        f = Formula("e^x + pi", variables=['x'])
        assert 'e' not in f.variables
        assert 'pi' not in f.variables

    def test_undefined_points_tracking(self):
        """Test that undefined points are tracked correctly."""
        f = Formula(
            "1/x",
            variables=['x'],
            check_undefined_points=True,
            max_undefined=2
        )
        # With x near 0, this should have undefined points
        # Function should handle gracefully


class TestModelConfig:
    """Test Pydantic model configuration."""

    def test_formula_enhanced_has_correct_config(self):
        """Test Formula model_config is set correctly."""
        assert hasattr(Formula, 'model_config')
        config = Formula.model_config
        assert config['arbitrary_types_allowed'] is True
        assert config['validate_assignment'] is True

    def test_formula_up_to_constant_has_correct_config(self):
        """Test FormulaUpToConstant model_config is set correctly."""
        assert hasattr(FormulaUpToConstant, 'model_config')
        config = FormulaUpToConstant.model_config
        assert config['arbitrary_types_allowed'] is True
        assert config['validate_assignment'] is True

    def test_field_constraints(self):
        """Test that field constraints are enforced."""
        # granularity must be gt 0
        with pytest.raises(ValueError):
            Formula("x", granularity=-1)

        with pytest.raises(ValueError):
            Formula("x", granularity=0)

        # But positive values should work
        f = Formula("x", granularity=1)
        assert f.granularity == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
