"""
Vector answer evaluator.

Handles vector comparison with component-wise tolerance.
"""

from __future__ import annotations

from typing import Any

from pg.math import Real, ToleranceMode, Vector
from pg.parser import Parser

from ..answer_hash import AnswerResult
from ..evaluator import AnswerEvaluator


class VectorEvaluator(AnswerEvaluator):
    """Evaluator for vector answers."""

    answer_type = "vector"

    def __init__(
        self,
        correct_answer: Vector | str | list[float],
        tolerance: float = 0.001,
        tolerance_mode: str = ToleranceMode.RELATIVE,
        **options: Any,
    ):
        """Initialize vector evaluator."""
        if isinstance(correct_answer, Vector):
            self.correct_vector = correct_answer
        elif isinstance(correct_answer, str):
            self.correct_vector = self._parse_vector(correct_answer)
        elif isinstance(correct_answer, list):
            self.correct_vector = Vector([Real(x) for x in correct_answer])
        else:
            raise TypeError(f"Invalid correct answer type: {type(correct_answer)}")

        super().__init__(
            self.correct_vector, tolerance, tolerance_mode, **options
        )
        self.parser = Parser()

    def _parse_vector(self, answer: str) -> Vector:
        """Parse vector from string."""
        try:
            from pg.parser.visitors import EvalVisitor

            ast = self.parser.parse(answer)
            visitor = EvalVisitor()
            result = ast.accept(visitor)

            if isinstance(result, tuple):
                return Vector([Real(x) for x in result])
            else:
                raise ValueError("Could not parse as vector")
        except Exception as e:
            raise ValueError(f"Could not parse vector: {e}") from e

    def parse_student_answer(self, answer: str) -> Vector:
        """Parse student's vector answer."""
        return self._parse_vector(answer)

    def compare(
        self, student_vector: Vector, correct_vector: Vector
    ) -> tuple[bool, float]:
        """Compare vectors component-wise."""
        is_correct = student_vector.compare(
            correct_vector, self.tolerance, self.tolerance_mode
        )
        return is_correct, 1.0 if is_correct else 0.0

    def evaluate(self, student_answer: str) -> AnswerResult:
        """Evaluate vector answer."""
        if not student_answer.strip():
            return AnswerResult.error_answer(
                student_answer, "Answer cannot be blank", self.answer_type
            )

        try:
            student_vector = self.parse_student_answer(student_answer)
        except ValueError as e:
            return AnswerResult.error_answer(
                student_answer, f"Could not parse vector: {e}", self.answer_type
            )

        is_correct, score = self.compare(student_vector, self.correct_vector)

        if is_correct:
            result = AnswerResult.correct_answer(
                student_vector.to_string(),
                self.correct_vector.to_string(),
                self.answer_type,
            )
        else:
            result = AnswerResult.incorrect_answer(
                student_vector.to_string(),
                self.correct_vector.to_string(),
                self.answer_type,
            )

        result.preview = student_vector.to_tex()
        result.original_student_answer = student_answer
        return result
