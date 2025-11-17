"""
Base answer evaluator framework.

Provides abstract base class for answer evaluators and a registry
for type-based dispatch.

Reference: lib/AnswerEvaluator.pm in legacy Perl codebase
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pg.math import MathValue

from .answer_hash import AnswerResult


class AnswerEvaluator(BaseModel, ABC):
    """
    Abstract base class for answer evaluators.

    Each evaluator is responsible for checking answers of a specific type
    (numeric, formula, string, etc.) against a correct answer.

    Pydantic-based with field validation for tolerance modes.

    Subclasses must implement:
    - evaluate(): Core evaluation logic
    - answer_type: Class variable for type identification
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    # Type identifier (must be set by subclasses)
    answer_type: ClassVar[str] = "unknown"

    correct_answer: Any = Field(description="The correct answer to compare against")
    tolerance: float = Field(default=0.001, gt=0, description="Tolerance for fuzzy comparison (default 0.001 = 0.1%)")
    tolerance_mode: str = Field(default="relative", description="Mode for tolerance: 'relative', 'absolute', or 'sigfigs'")
    options: dict = Field(default_factory=dict, description="Additional evaluator-specific options")

    @field_validator('tolerance_mode')
    @classmethod
    def validate_tolerance_mode(cls, v: str) -> str:
        """Validate that tolerance_mode is one of the allowed values."""
        valid_modes = {"relative", "absolute", "sigfigs"}
        if v not in valid_modes:
            raise ValueError(f"tolerance_mode must be one of {valid_modes}, got '{v}'")
        return v

    @abstractmethod
    def evaluate(self, student_answer: str) -> AnswerResult:
        """
        Evaluate student's answer against correct answer.

        Args:
            student_answer: Student's answer (as string input)

        Returns:
            AnswerResult with score, messages, etc.

        This method should:
        1. Parse/normalize student input
        2. Compare with correct answer
        3. Return AnswerResult with appropriate score and feedback
        """
        pass

    def parse_student_answer(self, answer: str) -> MathValue | str | Any:
        """
        Parse student's answer from string.

        Override this in subclasses to provide type-specific parsing.

        Args:
            answer: Raw student input

        Returns:
            Parsed answer value

        Raises:
            ValueError: If answer cannot be parsed
        """
        return answer

    def compare(
        self, student_value: Any, correct_value: Any
    ) -> tuple[bool, float]:
        """
        Compare student and correct values.

        Override this in subclasses for type-specific comparison.

        Args:
            student_value: Parsed student answer
            correct_value: Correct answer

        Returns:
            Tuple of (is_correct, score)
            - is_correct: Boolean correctness
            - score: Numeric score (0.0 to 1.0)
        """
        # Default: exact equality
        is_correct = student_value == correct_value
        return is_correct, 1.0 if is_correct else 0.0

    def get_preview(self, student_value: Any) -> str:
        """
        Generate preview (LaTeX/HTML) of student's answer.

        Override in subclasses for better formatting.

        Args:
            student_value: Parsed student answer

        Returns:
            Preview string (LaTeX or HTML)
        """
        return str(student_value)

    def get_correct_answer_display(self) -> str:
        """
        Get display string for correct answer.

        Returns:
            String representation of correct answer
        """
        return str(self.correct_answer)


class EvaluatorRegistry(BaseModel):
    """
    Registry for answer evaluators.

    Provides type-based dispatch to appropriate evaluator.
    Pydantic-based with private attribute management.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self) -> None:
        """Initialize empty registry."""
        super().__init__()
        self._evaluators: dict[str, type[AnswerEvaluator]] = {}

    def register(
        self, answer_type: str, evaluator_class: type[AnswerEvaluator]
    ) -> None:
        """
        Register an evaluator for a specific answer type.

        Args:
            answer_type: Type identifier (e.g., "numeric", "formula")
            evaluator_class: Evaluator class to use for this type

        Raises:
            TypeError: If evaluator_class is not an AnswerEvaluator subclass
        """
        if not (isinstance(evaluator_class, type) and issubclass(evaluator_class, AnswerEvaluator)):
            raise TypeError(f"evaluator_class must be a subclass of AnswerEvaluator, got {evaluator_class}")
        self._evaluators[answer_type] = evaluator_class

    def get_evaluator(self, answer_type: str) -> type[AnswerEvaluator] | None:
        """
        Get evaluator class for an answer type.

        Args:
            answer_type: Type identifier

        Returns:
            Evaluator class, or None if not found
        """
        return self._evaluators.get(answer_type)

    def create_evaluator(
        self,
        answer_type: str,
        correct_answer: Any,
        **options: Any,
    ) -> AnswerEvaluator:
        """
        Create evaluator instance for an answer type.

        Args:
            answer_type: Type identifier
            correct_answer: Correct answer
            **options: Evaluator-specific options

        Returns:
            Evaluator instance

        Raises:
            ValueError: If answer type not registered
        """
        evaluator_class = self.get_evaluator(answer_type)
        if evaluator_class is None:
            raise ValueError(f"No evaluator registered for type: {answer_type}")

        return evaluator_class(correct_answer=correct_answer, **options)

    def get_registered_types(self) -> list[str]:
        """
        Get list of all registered answer types.

        Returns:
            List of type identifiers
        """
        return list(self._evaluators.keys())


# Global registry instance
_global_registry = EvaluatorRegistry()


def register_evaluator(
    answer_type: str, evaluator_class: type[AnswerEvaluator]
) -> None:
    """
    Register an evaluator in the global registry.

    Args:
        answer_type: Type identifier
        evaluator_class: Evaluator class
    """
    _global_registry.register(answer_type, evaluator_class)


def get_evaluator(answer_type: str) -> type[AnswerEvaluator] | None:
    """
    Get evaluator from global registry.

    Args:
        answer_type: Type identifier

    Returns:
        Evaluator class or None
    """
    return _global_registry.get_evaluator(answer_type)


def create_evaluator(
    answer_type: str, correct_answer: Any, **options: Any
) -> AnswerEvaluator:
    """
    Create evaluator instance from global registry.

    Args:
        answer_type: Type identifier
        correct_answer: Correct answer
        **options: Evaluator-specific options

    Returns:
        Evaluator instance
    """
    return _global_registry.create_evaluator(answer_type, correct_answer, **options)
