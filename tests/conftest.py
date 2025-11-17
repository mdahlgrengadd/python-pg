"""
Shared pytest fixtures and utilities for testing Pydantic models in pg/math and pg/answer.

This module provides:
- Fixtures for comparing mathematical types
- Utilities for testing Pydantic validation
- Common test helpers for operators and serialization
"""

import pytest
from typing import Any, Type, TypeVar
from pydantic import BaseModel, ValidationError


T = TypeVar('T', bound=BaseModel)


@pytest.fixture
def pydantic_model_factory():
    """Factory for creating and validating Pydantic models in tests."""
    def _factory(model_class: Type[T], **kwargs: Any) -> T:
        """Create an instance of a Pydantic model, raising on validation error."""
        return model_class(**kwargs)
    return _factory


@pytest.fixture
def assert_validation_error():
    """Helper to assert that a ValidationError is raised with expected details."""
    def _assert_validation(
        model_class: Type[BaseModel],
        data: dict[str, Any],
        expected_field: str | None = None,
        expected_type: str | None = None,
    ) -> ValidationError:
        """
        Assert that creating a model raises ValidationError.

        Args:
            model_class: The Pydantic model class
            data: Invalid data to pass to model
            expected_field: Expected field name in error (optional)
            expected_type: Expected error type (optional)

        Returns:
            The ValidationError that was raised
        """
        with pytest.raises(ValidationError) as exc_info:
            model_class(**data)

        error = exc_info.value
        if expected_field:
            field_errors = [e for e in error.errors() if e['loc'][0] == expected_field]
            assert len(field_errors) > 0, f"Expected error for field '{expected_field}' not found"

        if expected_type:
            assert any(
                expected_type in str(e['type']).lower() for e in error.errors()
            ), f"Expected error type containing '{expected_type}' not found"

        return error

    return _assert_validation


@pytest.fixture
def assert_model_equality():
    """Helper to assert that two Pydantic models are equal."""
    def _assert_equal(model1: BaseModel, model2: BaseModel, ignore_fields: set[str] | None = None) -> None:
        """
        Assert that two models are equal, optionally ignoring certain fields.

        Args:
            model1: First model to compare
            model2: Second model to compare
            ignore_fields: Set of field names to ignore in comparison
        """
        ignore_fields = ignore_fields or set()

        dict1 = model1.model_dump(exclude=ignore_fields)
        dict2 = model2.model_dump(exclude=ignore_fields)

        assert dict1 == dict2, f"Models not equal:\n{dict1}\n!=\n{dict2}"

    return _assert_equal


@pytest.fixture
def assert_serializable():
    """Helper to assert that a model can be serialized and deserialized."""
    def _assert_serialization(
        model: BaseModel,
        model_class: Type[T],
        exclude_fields: set[str] | None = None,
    ) -> T:
        """
        Assert that a model can be serialized to dict and reconstructed.

        Args:
            model: The model instance to test
            model_class: The model class for reconstruction
            exclude_fields: Fields to exclude from serialization

        Returns:
            The reconstructed model
        """
        # Serialize to dict
        serialized = model.model_dump(exclude=exclude_fields or set())

        # Reconstruct from dict
        reconstructed = model_class(**serialized)

        # Verify round-trip
        assert reconstructed.model_dump(exclude=exclude_fields or set()) == serialized

        return reconstructed

    return _assert_serialization


# Marker for tests that verify no regression from original behavior
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]
