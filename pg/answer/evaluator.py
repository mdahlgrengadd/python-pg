"""
Base answer evaluator framework.

Provides abstract base class for answer evaluators and a registry
for type-based dispatch.

Reference: lib/AnswerEvaluator.pm in legacy Perl codebase
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pg.math import MathValue

from .answer_hash import AnswerResult


class AnswerEvaluator(ABC):
    """
    Abstract base class for answer evaluators.

    Each evaluator is responsible for checking answers of a specific type
    (numeric, formula, string, etc.) against a correct answer.

    Subclasses must implement:
    - evaluate(): Core evaluation logic
    - answer_type: Class variable for type identification
    """

    # Type identifier (must be set by subclasses)
    answer_type: ClassVar[str] = "unknown"

    def __init__(
        self,
        correct_answer: MathValue | str | Any,
        tolerance: float = 0.001,
        tolerance_mode: str = "relative",
        **options: Any,
    ):
        """
        Initialize evaluator with correct answer and options.

        Args:
            correct_answer: The correct answer to compare against
            tolerance: Tolerance for fuzzy comparison (default 0.001 = 0.1%)
            tolerance_mode: Mode for tolerance ("relative", "absolute", "sigfigs")
            **options: Additional evaluator-specific options
        """
        self.correct_answer = correct_answer
        self.tolerance = tolerance
        self.tolerance_mode = tolerance_mode
        self.options = options

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


class EvaluatorRegistry:
    """
    Registry for answer evaluators.

    Provides type-based dispatch to appropriate evaluator.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._evaluators: dict[str, type[AnswerEvaluator]] = {}

    def register(
        self, answer_type: str, evaluator_class: type[AnswerEvaluator]
    ) -> None:
        """
        Register an evaluator for a specific answer type.

        Args:
            answer_type: Type identifier (e.g., "numeric", "formula")
            evaluator_class: Evaluator class to use for this type
        """
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

        return evaluator_class(correct_answer, **options)

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
