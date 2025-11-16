"""
Matrix answer evaluator.

Handles matrix comparison with element-wise tolerance.
"""

from __future__ import annotations

from typing import Any

from pg.math import Matrix, Real, ToleranceMode
from pg.parser import Parser

from ..answer_hash import AnswerResult
from ..evaluator import AnswerEvaluator


class MatrixEvaluator(AnswerEvaluator):
    """Evaluator for matrix answers."""

    answer_type = "matrix"

    def __init__(
        self,
        correct_answer: Matrix | str | list[list[float]],
        tolerance: float = 0.001,
        tolerance_mode: str = ToleranceMode.RELATIVE,
        **options: Any,
    ):
        """Initialize matrix evaluator."""
        if isinstance(correct_answer, Matrix):
            self.correct_matrix = correct_answer
        elif isinstance(correct_answer, str):
            self.correct_matrix = self._parse_matrix(correct_answer)
        elif isinstance(correct_answer, list):
            self.correct_matrix = Matrix(
                [[Real(x) for x in row] for row in correct_answer]
            )
        else:
            raise TypeError(f"Invalid correct answer type: {type(correct_answer)}")

        super().__init__(
            self.correct_matrix, tolerance, tolerance_mode, **options
        )
        self.parser = Parser()

    def _parse_matrix(self, answer: str) -> Matrix:
        """Parse matrix from string."""
        try:
            from pg.parser.visitors import EvalVisitor

            ast = self.parser.parse(answer)
            visitor = EvalVisitor()
            result = ast.accept(visitor)

            if isinstance(result, list) and all(
                isinstance(row, list) for row in result
            ):
                return Matrix([[Real(x) for x in row] for row in result])
            else:
                raise ValueError("Could not parse as matrix")
        except Exception as e:
            raise ValueError(f"Could not parse matrix: {e}") from e

    def parse_student_answer(self, answer: str) -> Matrix:
        """Parse student's matrix answer."""
        return self._parse_matrix(answer)

    def compare(
        self, student_matrix: Matrix, correct_matrix: Matrix
    ) -> tuple[bool, float]:
        """Compare matrices element-wise."""
        is_correct = student_matrix.compare(
            correct_matrix, self.tolerance, self.tolerance_mode
        )
        return is_correct, 1.0 if is_correct else 0.0

    def evaluate(self, student_answer: str) -> AnswerResult:
        """Evaluate matrix answer."""
        if not student_answer.strip():
            return AnswerResult.error_answer(
                student_answer, "Answer cannot be blank", self.answer_type
            )

        try:
            student_matrix = self.parse_student_answer(student_answer)
        except ValueError as e:
            return AnswerResult.error_answer(
                student_answer, f"Could not parse matrix: {e}", self.answer_type
            )

        is_correct, score = self.compare(student_matrix, self.correct_matrix)

        if is_correct:
            result = AnswerResult.correct_answer(
                student_matrix.to_string(),
                self.correct_matrix.to_string(),
                self.answer_type,
            )
        else:
            result = AnswerResult.incorrect_answer(
                student_matrix.to_string(),
                self.correct_matrix.to_string(),
                self.answer_type,
            )

        result.preview = student_matrix.to_tex()
        result.original_student_answer = student_answer
        return result
