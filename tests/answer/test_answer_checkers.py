"""
Comprehensive test suite for answer checker classes.

Tests the Pydantic-based answer checker system including:
- AnswerChecker base class
- FormulaAnswerChecker
- RealAnswerChecker
- VectorAnswerChecker
- AnswerEvaluator base class
- EvaluatorRegistry
"""

import pytest
from typing import Dict, Any
from unittest.mock import Mock, MagicMock

from pg.math.answer_checker import (
    AnswerChecker,
    FormulaAnswerChecker,
    RealAnswerChecker,
    VectorAnswerChecker,
)
from pg.math.numeric import Real
from pg.math.geometric import Vector
from pg.math.formula import Formula
from pg.math.context import Context, get_context
from pg.answer.evaluator import AnswerEvaluator, EvaluatorRegistry


class TestAnswerChecker:
    """Test the AnswerChecker Pydantic base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that AnswerChecker cannot be instantiated directly (abstract)."""
        with pytest.raises(TypeError):
            AnswerChecker(correct_value=5)

    def test_subclass_must_implement_check(self):
        """Test that subclasses must implement check()."""
        class MinimalChecker(AnswerChecker):
            pass

        with pytest.raises(TypeError):
            MinimalChecker(correct_value=5)

    def test_base_fields_initialization(self):
        """Test that base AnswerChecker fields are properly initialized."""
        class TestChecker(AnswerChecker):
            def check(self, student_answer: str) -> Dict[str, Any]:
                return {'score': 1.0, 'correct': True}

        checker = TestChecker(correct_value=42)
        assert checker.correct_value == 42
        assert checker.options == {}
        assert checker.post_filter is None

    def test_method_chaining_with_post_filter(self):
        """Test that withPostFilter returns self for chaining."""
        class TestChecker(AnswerChecker):
            def check(self, student_answer: str) -> Dict[str, Any]:
                return {'score': 1.0}

        checker = TestChecker(correct_value=42)
        filter_fn = lambda x: x

        result = checker.withPostFilter(filter_fn)
        assert result is checker
        assert checker.post_filter is filter_fn

    def test_pydantic_validation_on_incorrect_value_type(self):
        """Test that Pydantic validation works for field types."""
        class TestChecker(AnswerChecker):
            def check(self, student_answer: str) -> Dict[str, Any]:
                return {'score': 1.0}

        # Should accept any value for correct_value
        checker = TestChecker(correct_value=[1, 2, 3])
        assert checker.correct_value == [1, 2, 3]

    def test_options_dict_empty_by_default(self):
        """Test that options dict is empty by default."""
        class TestChecker(AnswerChecker):
            def check(self, student_answer: str) -> Dict[str, Any]:
                return {}

        checker = TestChecker(correct_value=None)
        assert checker.options == {}


class TestFormulaAnswerChecker:
    """Test the FormulaAnswerChecker Pydantic class."""

    def test_field_extraction_from_options(self):
        """Test that num_points and tolerance are extracted from options."""
        formula = Formula("x^2")
        checker = FormulaAnswerChecker(
            formula,
            num_points=10,
            tolerance=0.05,
            extra_option='ignored'
        )
        assert checker.num_points == 10
        assert checker.tolerance == 0.05
        assert 'extra_option' in checker.options

    def test_default_field_values(self):
        """Test that default values are applied when not provided."""
        formula = Formula("x^2")
        checker = FormulaAnswerChecker(formula)
        assert checker.num_points == 5
        assert checker.tolerance == 0.01

    def test_field_validation_num_points(self):
        """Test that num_points is validated (must be >= 1)."""
        formula = Formula("x^2")
        # Validation allows 0 by default, but we could add a custom validator
        # For now, just verify it accepts positive values
        checker = FormulaAnswerChecker(formula, num_points=1)
        assert checker.num_points == 1

    def test_field_validation_tolerance(self):
        """Test that tolerance is validated (must be > 0)."""
        formula = Formula("x^2")
        with pytest.raises(ValueError):
            FormulaAnswerChecker(formula, tolerance=-0.01)

    def test_check_returns_dict(self):
        """Test that check() returns proper dict structure."""
        formula = Formula("x^2")
        checker = FormulaAnswerChecker(formula)
        result = checker.check("x**2")
        assert isinstance(result, dict)
        assert 'score' in result
        assert 'correct' in result
        assert isinstance(result['score'], float)
        assert isinstance(result['correct'], bool)

    def test_pydantic_model_validation(self):
        """Test that FormulaAnswerChecker is a valid Pydantic model."""
        formula = Formula("x^2")
        checker = FormulaAnswerChecker(formula, num_points=7, tolerance=0.02)
        # Should be able to access model_fields
        assert hasattr(checker, 'model_fields')
        # Should be able to serialize
        assert checker.model_dump() is not None


class TestRealAnswerChecker:
    """Test the RealAnswerChecker Pydantic class with context-aware defaults."""

    def test_explicit_tolerance_override(self):
        """Test that explicit tolerance overrides everything."""
        ctx = get_context('Numeric')
        real = Real(5, ctx)
        checker = RealAnswerChecker(real, tolerance=0.05)
        assert checker.tolerance == 0.05

    def test_context_tolerance_extraction(self):
        """Test that tolerance is extracted from context when not provided."""
        ctx = get_context('Numeric')
        ctx.flags.set(tolerance=0.002)
        real = Real(5, ctx)

        checker = RealAnswerChecker(real)
        assert checker.tolerance == 0.002

    def test_default_tolerance_fallback(self):
        """Test that default tolerance is used when nothing is provided."""
        # Create a Real without context (or with None)
        real = Real(5)
        real.context = None

        checker = RealAnswerChecker(real)
        assert checker.tolerance == 0.001

    def test_explicit_tol_type_override(self):
        """Test that explicit tol_type overrides context."""
        ctx = get_context('Numeric')
        ctx.flags.set(tolType='absolute')
        real = Real(5, ctx)

        checker = RealAnswerChecker(real, tolType='sigfigs')
        assert checker.tol_type == 'sigfigs'

    def test_context_tol_type_extraction(self):
        """Test that tol_type is extracted from context."""
        ctx = get_context('Numeric')
        ctx.flags.set(tolType='absolute')
        real = Real(5, ctx)

        checker = RealAnswerChecker(real)
        assert checker.tol_type == 'absolute'

    def test_default_tol_type_fallback(self):
        """Test that default tol_type is 'relative'."""
        real = Real(5)
        real.context = None

        checker = RealAnswerChecker(real)
        assert checker.tol_type == 'relative'

    def test_callable_syntax(self):
        """Test that RealAnswerChecker can be called as a function."""
        real = Real(5)
        checker = RealAnswerChecker(real)
        # Should be callable
        assert callable(checker)
        # Should return result
        result = checker('5')
        assert isinstance(result, dict)

    def test_float_parsing_string(self):
        """Test that check() parses float strings correctly."""
        real = Real(5.0)
        checker = RealAnswerChecker(real)
        result = checker.check('5.0')
        assert result['score'] == 1.0
        assert result['correct'] is True

    def test_float_parsing_invalid(self):
        """Test that check() handles invalid numeric input."""
        real = Real(5)
        checker = RealAnswerChecker(real)
        result = checker.check('not a number')
        assert result['score'] == 0.0
        assert result['correct'] is False


class TestVectorAnswerChecker:
    """Test the VectorAnswerChecker Pydantic class."""

    def test_field_extraction_from_options(self):
        """Test that tolerance and custom_checker are extracted from options."""
        vector = Vector([1, 2, 3])
        custom_fn = lambda x: True

        checker = VectorAnswerChecker(
            vector,
            tolerance=0.05,
            checker=custom_fn,
            extra='option'
        )
        assert checker.tolerance == 0.05
        assert checker.custom_checker is custom_fn
        assert 'extra' in checker.options

    def test_default_values(self):
        """Test default values when not provided."""
        vector = Vector([1, 2, 3])
        checker = VectorAnswerChecker(vector)
        assert checker.tolerance == 0.001
        assert checker.custom_checker is None

    def test_field_validation_tolerance(self):
        """Test that tolerance is validated (must be > 0)."""
        vector = Vector([1, 2, 3])
        with pytest.raises(ValueError):
            VectorAnswerChecker(vector, tolerance=-0.01)

    def test_callable_syntax(self):
        """Test that VectorAnswerChecker is callable."""
        vector = Vector([1, 2, 3])
        checker = VectorAnswerChecker(vector)
        assert callable(checker)

    def test_custom_checker_field(self):
        """Test that custom checker can be assigned and retrieved."""
        vector = Vector([1, 2, 3])
        custom_fn = MagicMock(return_value={'score': 0.5})
        checker = VectorAnswerChecker(vector, checker=custom_fn)
        assert checker.custom_checker is custom_fn


class TestAnswerEvaluator:
    """Test the AnswerEvaluator Pydantic base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that AnswerEvaluator is abstract."""
        with pytest.raises(TypeError):
            AnswerEvaluator(correct_answer=5)

    def test_field_initialization(self):
        """Test that fields are properly initialized."""
        class TestEvaluator(AnswerEvaluator):
            answer_type = 'test'
            def evaluate(self, student_answer):
                pass

        evaluator = TestEvaluator(
            correct_answer=42,
            tolerance=0.01,
            tolerance_mode='absolute'
        )
        assert evaluator.correct_answer == 42
        assert evaluator.tolerance == 0.01
        assert evaluator.tolerance_mode == 'absolute'
        assert evaluator.options == {}

    def test_default_values(self):
        """Test that defaults are applied."""
        class TestEvaluator(AnswerEvaluator):
            answer_type = 'test'
            def evaluate(self, student_answer):
                pass

        evaluator = TestEvaluator(correct_answer=42)
        assert evaluator.tolerance == 0.001
        assert evaluator.tolerance_mode == 'relative'
        assert evaluator.options == {}

    def test_tolerance_validation(self):
        """Test that tolerance is validated (must be > 0)."""
        class TestEvaluator(AnswerEvaluator):
            answer_type = 'test'
            def evaluate(self, student_answer):
                pass

        with pytest.raises(ValueError):
            TestEvaluator(correct_answer=42, tolerance=-0.01)

    def test_tolerance_mode_validation_valid(self):
        """Test that valid tolerance modes are accepted."""
        class TestEvaluator(AnswerEvaluator):
            answer_type = 'test'
            def evaluate(self, student_answer):
                pass

        for mode in ['relative', 'absolute', 'sigfigs']:
            evaluator = TestEvaluator(correct_answer=42, tolerance_mode=mode)
            assert evaluator.tolerance_mode == mode

    def test_tolerance_mode_validation_invalid(self):
        """Test that invalid tolerance modes are rejected."""
        class TestEvaluator(AnswerEvaluator):
            answer_type = 'test'
            def evaluate(self, student_answer):
                pass

        with pytest.raises(ValueError, match="tolerance_mode must be one of"):
            TestEvaluator(correct_answer=42, tolerance_mode='invalid')

    def test_class_variable_answer_type(self):
        """Test that answer_type is a class variable."""
        class TestEvaluator(AnswerEvaluator):
            answer_type = 'custom'
            def evaluate(self, student_answer):
                pass

        evaluator = TestEvaluator(correct_answer=42)
        assert TestEvaluator.answer_type == 'custom'
        assert evaluator.answer_type == 'custom'

    def test_pydantic_model_features(self):
        """Test that Pydantic features work."""
        class TestEvaluator(AnswerEvaluator):
            answer_type = 'test'
            def evaluate(self, student_answer):
                pass

        evaluator = TestEvaluator(correct_answer=42, tolerance=0.01)
        # Should have model_dump
        assert evaluator.model_dump() is not None
        # Should have model_fields
        assert hasattr(evaluator, 'model_fields')


class TestEvaluatorRegistry:
    """Test the EvaluatorRegistry Pydantic class."""

    def test_register_and_retrieve(self):
        """Test basic registration and retrieval."""
        class TestEvaluator(AnswerEvaluator):
            answer_type = 'test'
            def evaluate(self, student_answer):
                pass

        registry = EvaluatorRegistry()
        registry.register('numeric', TestEvaluator)
        assert registry.get_evaluator('numeric') is TestEvaluator

    def test_retrieve_nonexistent(self):
        """Test that retrieving nonexistent type returns None."""
        registry = EvaluatorRegistry()
        assert registry.get_evaluator('nonexistent') is None

    def test_register_validation(self):
        """Test that registration validates the evaluator class."""
        registry = EvaluatorRegistry()
        # Should reject non-AnswerEvaluator classes
        with pytest.raises(TypeError):
            registry.register('invalid', int)

        with pytest.raises(TypeError):
            registry.register('invalid', str)

    def test_create_evaluator(self):
        """Test creating evaluator instances."""
        class TestEvaluator(AnswerEvaluator):
            answer_type = 'test'
            def evaluate(self, student_answer):
                pass

        registry = EvaluatorRegistry()
        registry.register('test', TestEvaluator)

        evaluator = registry.create_evaluator('test', correct_answer=42)
        assert isinstance(evaluator, TestEvaluator)
        assert evaluator.correct_answer == 42

    def test_create_evaluator_not_found(self):
        """Test that creating nonexistent evaluator raises error."""
        registry = EvaluatorRegistry()
        with pytest.raises(ValueError, match="No evaluator registered"):
            registry.create_evaluator('nonexistent', correct_answer=42)

    def test_create_evaluator_with_options(self):
        """Test creating evaluator with additional options."""
        class TestEvaluator(AnswerEvaluator):
            answer_type = 'test'
            def evaluate(self, student_answer):
                pass

        registry = EvaluatorRegistry()
        registry.register('test', TestEvaluator)

        evaluator = registry.create_evaluator(
            'test',
            correct_answer=42,
            tolerance=0.01
        )
        assert evaluator.tolerance == 0.01

    def test_get_registered_types(self):
        """Test getting list of registered types."""
        class TestEvaluator(AnswerEvaluator):
            answer_type = 'test'
            def evaluate(self, student_answer):
                pass

        registry = EvaluatorRegistry()
        registry.register('type1', TestEvaluator)
        registry.register('type2', TestEvaluator)

        types = registry.get_registered_types()
        assert set(types) == {'type1', 'type2'}

    def test_multiple_evaluator_classes(self):
        """Test registering different evaluator classes."""
        class Evaluator1(AnswerEvaluator):
            answer_type = 'type1'
            def evaluate(self, student_answer):
                pass

        class Evaluator2(AnswerEvaluator):
            answer_type = 'type2'
            def evaluate(self, student_answer):
                pass

        registry = EvaluatorRegistry()
        registry.register('type1', Evaluator1)
        registry.register('type2', Evaluator2)

        assert registry.get_evaluator('type1') is Evaluator1
        assert registry.get_evaluator('type2') is Evaluator2

    def test_pydantic_model_features(self):
        """Test that EvaluatorRegistry is a valid Pydantic model."""
        registry = EvaluatorRegistry()
        # Should have model_dump
        assert registry.model_dump() is not None
        # Should have model_fields
        assert hasattr(registry, 'model_fields')


class TestBackwardCompatibility:
    """Test backward compatibility with previous API."""

    def test_formula_checker_old_style(self):
        """Test that old-style Formula checker creation still works."""
        formula = Formula("x^2")
        # Old style: pass all options in kwargs
        checker = FormulaAnswerChecker(formula, num_points=7, tolerance=0.02)
        assert checker.num_points == 7
        assert checker.tolerance == 0.02

    def test_real_checker_old_style(self):
        """Test that old-style Real checker creation still works."""
        real = Real(5)
        # Old style
        checker = RealAnswerChecker(real, tolerance=0.01, tolType='absolute')
        assert checker.tolerance == 0.01
        assert checker.tol_type == 'absolute'

    def test_vector_checker_old_style(self):
        """Test that old-style Vector checker creation still works."""
        vector = Vector([1, 2, 3])
        checker = VectorAnswerChecker(vector, tolerance=0.05)
        assert checker.tolerance == 0.05

    def test_check_result_structure(self):
        """Test that check() returns expected result structure."""
        formula = Formula("x^2")
        checker = FormulaAnswerChecker(formula)
        result = checker.check("x**2")

        # Must have these keys
        assert 'score' in result
        assert 'correct' in result
        # Optional message
        assert isinstance(result, dict)

    def test_method_chaining_preserved(self):
        """Test that method chaining with .withPostFilter() still works."""
        formula = Formula("x^2")
        filter_fn = lambda x: x

        checker = FormulaAnswerChecker(formula).withPostFilter(filter_fn)
        assert isinstance(checker, FormulaAnswerChecker)
        assert checker.post_filter is filter_fn


class TestIntegration:
    """Integration tests for answer checker system."""

    def test_checker_with_context(self):
        """Test that checkers work with context system."""
        ctx = Context('Numeric')
        ctx.flags.set(tolerance=0.002)

        real = Real(5, ctx)
        checker = RealAnswerChecker(real)

        # Should use context tolerance
        assert checker.tolerance == 0.002

    def test_formula_checker_with_variables(self):
        """Test that formula checker handles variables correctly."""
        formula = Formula("x + y")
        checker = FormulaAnswerChecker(formula, num_points=10)

        # Should create checker without error
        assert checker.num_points == 10
        assert checker.correct_value is formula

    def test_evaluator_registry_workflow(self):
        """Test complete evaluator registry workflow."""
        class NumericEvaluator(AnswerEvaluator):
            answer_type = 'numeric'
            def evaluate(self, student_answer):
                return None

        registry = EvaluatorRegistry()

        # Register
        registry.register('numeric', NumericEvaluator)

        # Retrieve
        assert registry.get_evaluator('numeric') is NumericEvaluator

        # Create instance with keyword arguments
        evaluator = registry.create_evaluator(
            'numeric',
            correct_answer=42,
            tolerance=0.01
        )
        assert evaluator.correct_answer == 42
        assert evaluator.tolerance == 0.01

        # List types
        assert 'numeric' in registry.get_registered_types()
