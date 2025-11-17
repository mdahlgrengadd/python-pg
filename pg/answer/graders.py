"""
Problem graders for scoring multiple answer blanks.

Graders combine scores from multiple answer evaluators into a final
problem score.

Reference: lib/AnswerHash.pm grading methods (lines 400-500)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .answer_hash import AnswerResult


class Grader(BaseModel, ABC):
    """
    Abstract base class for problem graders.

    Pydantic-based graders take a list of AnswerResults (one per answer blank)
    and compute an overall problem score.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @abstractmethod
    def grade(self, answers: list[AnswerResult]) -> float:
        """
        Compute overall score from individual answer results.

        Args:
            answers: List of AnswerResult objects, one per answer blank

        Returns:
            Overall problem score (0.0 to 1.0)
        """
        pass


class StandardGrader(Grader):
    """
    Standard (all-or-nothing) grader.

    Problem is correct only if ALL answer blanks are correct.
    Score is 1.0 if all correct, 0.0 otherwise.
    """

    def grade(self, answers: list[AnswerResult]) -> float:
        """
        Grade using all-or-nothing strategy.

        Args:
            answers: List of answer results

        Returns:
            1.0 if all answers correct, 0.0 otherwise
        """
        if not answers:
            return 0.0

        # All answers must be correct
        all_correct = all(ans.correct for ans in answers)
        return 1.0 if all_correct else 0.0


class AverageGrader(Grader):
    """
    Average grader (supports partial credit).

    Problem score is the average of individual answer scores.
    This allows partial credit for getting some answers right.
    Pydantic-based with weights validation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    weights: Optional[list[float]] = Field(
        default=None,
        description="Optional weights for each answer (must sum to 1.0)"
    )

    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        """Validate that weights are positive and will be checked in grade()."""
        if v is not None:
            if not all(w >= 0 for w in v):
                raise ValueError("All weights must be non-negative")
        return v

    def __init__(self, weights: Optional[list[float]] = None, **kwargs):
        """
        Initialize average grader.

        Args:
            weights: Optional weights for each answer (must sum to 1.0)
                    If None, all answers weighted equally
        """
        super().__init__(weights=weights, **kwargs)

    def grade(self, answers: list[AnswerResult]) -> float:
        """
        Grade using weighted or unweighted average.

        Args:
            answers: List of answer results

        Returns:
            Average score (0.0 to 1.0)
        """
        if not answers:
            return 0.0

        if self.weights is not None:
            # Weighted average
            if len(self.weights) != len(answers):
                raise ValueError(
                    f"Number of weights ({len(self.weights)}) must match "
                    f"number of answers ({len(answers)})"
                )

            # Ensure weights sum to 1.0
            weight_sum = sum(self.weights)
            if abs(weight_sum - 1.0) > 0.001:
                raise ValueError(
                    f"Weights must sum to 1.0 (got {weight_sum})"
                )

            # Compute weighted average
            total = sum(
                ans.score * weight
                for ans, weight in zip(answers, self.weights)
            )
            return total
        else:
            # Unweighted average
            total = sum(ans.score for ans in answers)
            return total / len(answers)


class FirstAnswerGrader(Grader):
    """
    First answer grader.

    Problem score is based only on the first answer blank.
    Useful for problems with one primary answer and auxiliary blanks.
    Pydantic-based grader.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def grade(self, answers: list[AnswerResult]) -> float:
        """
        Grade using only the first answer.

        Args:
            answers: List of answer results

        Returns:
            Score of first answer (0.0 to 1.0)
        """
        if not answers:
            return 0.0

        return answers[0].score


class MinimumGrader(Grader):
    """
    Minimum grader.

    Problem score is the minimum of all answer scores.
    Useful when all parts are equally critical.
    Pydantic-based grader.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def grade(self, answers: list[AnswerResult]) -> float:
        """
        Grade using minimum score.

        Args:
            answers: List of answer results

        Returns:
            Minimum score (0.0 to 1.0)
        """
        if not answers:
            return 0.0

        return min(ans.score for ans in answers)


class CustomGrader(Grader):
    """
    Custom grader using user-provided function.

    Allows arbitrary grading logic via a callback function.
    Pydantic-based with callable field.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    grading_function: Optional[Any] = Field(
        default=None,
        description="Function taking list[AnswerResult] → float"
    )

    def __init__(self, grading_function: Optional[Any] = None, **kwargs):
        """
        Initialize custom grader.

        Args:
            grading_function: Function taking list[AnswerResult] → float
        """
        super().__init__(grading_function=grading_function, **kwargs)

    def grade(self, answers: list[AnswerResult]) -> float:
        """
        Grade using custom function.

        Args:
            answers: List of answer results

        Returns:
            Score from custom function (0.0 to 1.0)
        """
        return self.grading_function(answers)
