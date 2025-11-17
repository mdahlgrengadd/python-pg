"""
Tests for Pydantic BaseModel compatibility in InProcessSandbox.

Verifies that Pydantic's BaseModel, Field, validators, and ConfigDict
work correctly within the restricted sandbox environment.
"""

import pytest
from pg.translator.in_process_sandbox import InProcessSandbox


class TestSandboxPydanticBasics:
    """Test basic Pydantic functionality in sandbox."""

    def test_pydantic_import_allowed(self):
        """Test that pydantic imports are allowed in sandbox."""
        sandbox = InProcessSandbox()
        code = """
from pydantic import BaseModel
result = "pydantic imported successfully"
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"Failed to import pydantic: {result.errors}"

    def test_basemodel_class_creation(self):
        """Test that BaseModel subclasses can be created in sandbox."""
        sandbox = InProcessSandbox()
        code = """
DOCUMENT()

from pydantic import BaseModel, Field

class TestModel(BaseModel):
    value: int = Field(default=1, gt=0)

instance = TestModel(value=42)
TEXT(f"BaseModel instance created: {instance.value}")

ENDDOCUMENT()
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"Failed to create BaseModel: {result.errors}"
        assert "instance created" in result.output_text

    def test_field_validators(self):
        """Test that field validators work in sandbox."""
        sandbox = InProcessSandbox()
        code = """
DOCUMENT()

from pydantic import BaseModel, Field, field_validator

class TestModel(BaseModel):
    value: int = Field(default=1)

    @field_validator("value")
    @classmethod
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("value must be positive")
        return v

instance = TestModel(value=42)
TEXT(f"Validation passed: {instance.value}")

ENDDOCUMENT()
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"Validator test failed: {result.errors}"
        assert "Validation passed" in result.output_text

    def test_field_validators_reject_invalid(self):
        """Test that validators correctly reject invalid values."""
        sandbox = InProcessSandbox()
        code = """
DOCUMENT()

from pydantic import BaseModel, Field, field_validator, ValidationError

class TestModel(BaseModel):
    value: int = Field(default=1)

    @field_validator("value")
    @classmethod
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("value must be positive")
        return v

try:
    instance = TestModel(value=-5)
    TEXT("UNEXPECTED: validation did not fail")
except ValidationError as e:
    TEXT(f"Validation correctly failed")

ENDDOCUMENT()
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"Test failed: {result.errors}"
        assert "correctly failed" in result.output_text

    def test_configdict_validate_assignment(self):
        """Test that ConfigDict with validate_assignment works."""
        sandbox = InProcessSandbox()
        code = """
DOCUMENT()

from pydantic import BaseModel, ConfigDict, Field

class TestModel(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    value: int = Field(default=1, gt=0)

instance = TestModel(value=10)
# Assign new value - should trigger validation
instance.value = 20
TEXT(f"Assignment validation works: {instance.value}")

ENDDOCUMENT()
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"ConfigDict test failed: {result.errors}"
        assert "Assignment validation works" in result.output_text

    def test_arbitrary_types_allowed(self):
        """Test that ConfigDict with arbitrary_types_allowed works."""
        sandbox = InProcessSandbox()
        code = """
DOCUMENT()

from pydantic import BaseModel, ConfigDict
from pg.math import Real

class TestModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    value: object

instance = TestModel(value=Real(5.0))
TEXT(f"Arbitrary types work: {instance.value}")

ENDDOCUMENT()
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"Arbitrary types test failed: {result.errors}"
        assert "Arbitrary types work" in result.output_text


class TestSandboxPydanticEvaluators:
    """Test Pydantic-based evaluators in sandbox."""

    @pytest.mark.xfail(reason="Evaluators imported from outside sandbox still access full __builtins__")
    def test_numeric_evaluator_construction(self):
        """Test that NumericEvaluator can be constructed in sandbox.

        NOTE: This test is expected to fail because NumericEvaluator is a Pydantic class
        that's imported from outside the sandbox. When it tries to validate_assignment,
        it still accesses the real Python __builtins__ where the class was defined,
        not the sandbox's restricted __builtins__.

        Solution: Pre-construct evaluators outside sandbox and inject them.
        """
        sandbox = InProcessSandbox()
        code = """
DOCUMENT()

from pg.math import Real
from pg.answer.evaluators.numeric import NumericEvaluator

evaluator = NumericEvaluator(correct_answer=Real(5.0))
TEXT(f"NumericEvaluator created: {evaluator.correct_answer}")

ENDDOCUMENT()
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"Failed to create NumericEvaluator: {result.errors}"
        assert "NumericEvaluator created" in result.output_text

    @pytest.mark.xfail(reason="Evaluators imported from outside sandbox still access full __builtins__")
    def test_formula_evaluator_construction(self):
        """Test that FormulaEvaluator can be constructed in sandbox.

        NOTE: This test is expected to fail for the same reason as test_numeric_evaluator_construction.
        """
        sandbox = InProcessSandbox()
        code = """
DOCUMENT()

from pg.math import Formula
from pg.answer.evaluators.formula import FormulaEvaluator

formula = Formula("x^2 + 1")
evaluator = FormulaEvaluator(correct_answer=formula)
TEXT(f"FormulaEvaluator created: {evaluator.correct_answer}")

ENDDOCUMENT()
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"Failed to create FormulaEvaluator: {result.errors}"
        assert "FormulaEvaluator created" in result.output_text

    @pytest.mark.xfail(reason="Evaluators imported from outside sandbox still access full __builtins__")
    def test_evaluator_with_options(self):
        """Test evaluator with configuration options.

        NOTE: This test is expected to fail for the same reason as test_numeric_evaluator_construction.
        """
        sandbox = InProcessSandbox()
        code = """
DOCUMENT()

from pg.math import Real
from pg.answer.evaluators.numeric import NumericEvaluator

evaluator = NumericEvaluator(
    correct_answer=Real(5.0),
    tolerance=0.1,
    allow_expressions=True
)
TEXT(f"Evaluator with options: tolerance={evaluator.tolerance}")

ENDDOCUMENT()
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"Failed to create evaluator with options: {result.errors}"
        assert "tolerance=0.1" in result.output_text


class TestSandboxPydanticDeclarators:
    """Test Pydantic class decorators and descriptors in sandbox."""

    def test_property_descriptor(self):
        """Test that @property decorator works."""
        sandbox = InProcessSandbox()
        code = """
DOCUMENT()

class TestClass:
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value * 2

instance = TestClass(5)
TEXT(f"Property works: {instance.value}")

ENDDOCUMENT()
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"Property test failed: {result.errors}"
        assert "Property works: 10" in result.output_text

    def test_classmethod_descriptor(self):
        """Test that @classmethod decorator works."""
        sandbox = InProcessSandbox()
        code = """
DOCUMENT()

class TestClass:
    count = 0

    @classmethod
    def create(cls):
        cls.count += 1
        return cls()

instance1 = TestClass.create()
instance2 = TestClass.create()
TEXT(f"Classmethod works: {TestClass.count}")

ENDDOCUMENT()
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"Classmethod test failed: {result.errors}"
        assert "Classmethod works: 2" in result.output_text

    def test_staticmethod_descriptor(self):
        """Test that @staticmethod decorator works."""
        sandbox = InProcessSandbox()
        code = """
DOCUMENT()

class TestClass:
    @staticmethod
    def add(a, b):
        return a + b

TEXT(f"Staticmethod works: {TestClass.add(2, 3)}")

ENDDOCUMENT()
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"Staticmethod test failed: {result.errors}"
        assert "Staticmethod works: 5" in result.output_text

    def test_super_inheritance(self):
        """Test that super() works with inheritance."""
        sandbox = InProcessSandbox()
        code = """
DOCUMENT()

class Parent:
    def __init__(self, value):
        self.value = value

class Child(Parent):
    def __init__(self, value, extra):
        super().__init__(value)
        self.extra = extra

instance = Child(5, 10)
TEXT(f"Super works: {instance.value}, {instance.extra}")

ENDDOCUMENT()
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"Super test failed: {result.errors}"
        assert "Super works: 5, 10" in result.output_text


class TestSandboxBuiltins:
    """Test that new builtins work correctly."""

    def test_iter_next_builtins(self):
        """Test iter() and next() builtins."""
        sandbox = InProcessSandbox()
        code = """
DOCUMENT()

items = [1, 2, 3]
it = iter(items)
first = next(it)
second = next(it)
TEXT(f"iter/next work: {first}, {second}")

ENDDOCUMENT()
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"iter/next test failed: {result.errors}"
        assert "iter/next work: 1, 2" in result.output_text

    def test_frozenset_builtin(self):
        """Test frozenset builtin."""
        sandbox = InProcessSandbox()
        code = """
DOCUMENT()

fs = frozenset([1, 2, 3])
TEXT(f"frozenset works: {len(fs)}")

ENDDOCUMENT()
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"frozenset test failed: {result.errors}"
        assert "frozenset works: 3" in result.output_text

    def test_repr_builtin(self):
        """Test repr() builtin."""
        sandbox = InProcessSandbox()
        code = """
DOCUMENT()

s = repr("hello")
TEXT(f"repr works: {s}")

ENDDOCUMENT()
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"repr test failed: {result.errors}"
        assert "repr works:" in result.output_text

    def test_format_builtin(self):
        """Test format() builtin."""
        sandbox = InProcessSandbox()
        code = """
DOCUMENT()

f = format(42, "d")
TEXT(f"format works: {f}")

ENDDOCUMENT()
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"format test failed: {result.errors}"
        assert "format works: 42" in result.output_text

    def test_delattr_builtin(self):
        """Test delattr() builtin."""
        sandbox = InProcessSandbox()
        code = """
DOCUMENT()

class TestClass:
    def __init__(self):
        self.value = 10

instance = TestClass()
delattr(instance, 'value')
TEXT(f"delattr works: {not hasattr(instance, 'value')}")

ENDDOCUMENT()
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"delattr test failed: {result.errors}"
        assert "delattr works: True" in result.output_text


class TestSandboxNameAttribute:
    """Test that __name__ attribute is available."""

    def test_module_name_available(self):
        """Test that __name__ is available in namespace."""
        sandbox = InProcessSandbox()
        code = """
DOCUMENT()

TEXT(f"Module name: {__name__}")

ENDDOCUMENT()
"""
        result = sandbox.execute(code, seed=1234)
        assert result.success, f"__name__ test failed: {result.errors}"
        assert "__sandbox__" in result.output_text
