"""
Interval answer evaluator.

Handles interval comparison with endpoint tolerance.
"""

from __future__ import annotations

from typing import Any

from pg.math import Interval, MathValue, Real, ToleranceMode
from pg.parser import Parser

from ..answer_hash import AnswerResult
from ..evaluator import AnswerEvaluator


class IntervalEvaluator(AnswerEvaluator):
    """Evaluator for interval answers."""

    answer_type = "interval"

    def __init__(
        self,
        correct_answer: Interval | str,
        tolerance: float = 0.001,
        tolerance_mode: str = ToleranceMode.RELATIVE,
        **options: Any,
    ):
        """Initialize interval evaluator."""
        if isinstance(correct_answer, Interval):
            self.correct_interval = correct_answer
        elif isinstance(correct_answer, str):
            self.correct_interval = self._parse_interval(correct_answer)
        else:
            raise TypeError(f"Invalid correct answer type: {type(correct_answer)}")

        super().__init__(
            self.correct_interval, tolerance, tolerance_mode, **options
        )
        self.parser = Parser()

    def _parse_interval(self, answer: str) -> Interval:
        """Parse interval from string."""
        try:
            from pg.parser.visitors import EvalVisitor

            ast = self.parser.parse(answer)
            visitor = EvalVisitor()
            result = ast.accept(visitor)

            if isinstance(result, tuple) and len(result) == 4:
                # (left, right, open_left, open_right)
                left, right, open_left, open_right = result
                return Interval(
                    Real(left), Real(right), open_left, open_right
                )
            else:
                raise ValueError("Could not parse as interval")
        except Exception as e:
            raise ValueError(f"Could not parse interval: {e}") from e

    def parse_student_answer(self, answer: str) -> Interval:
        """Parse student's interval answer."""
        return self._parse_interval(answer)

    def compare(
        self, student_interval: Interval, correct_interval: Interval
    ) -> tuple[bool, float]:
        """Compare intervals."""
        is_correct = student_interval.compare(
            correct_interval, self.tolerance, self.tolerance_mode
        )
        return is_correct, 1.0 if is_correct else 0.0

    def evaluate(self, student_answer: str) -> AnswerResult:
        """Evaluate interval answer."""
        if not student_answer.strip():
            return AnswerResult.error_answer(
                student_answer, "Answer cannot be blank", self.answer_type
            )

        try:
            student_interval = self.parse_student_answer(student_answer)
        except ValueError as e:
            return AnswerResult.error_answer(
                student_answer, f"Could not parse interval: {e}", self.answer_type
            )

        is_correct, score = self.compare(student_interval, self.correct_interval)

        if is_correct:
            result = AnswerResult.correct_answer(
                student_interval.to_string(),
                self.correct_interval.to_string(),
                self.answer_type,
            )
        else:
            result = AnswerResult.incorrect_answer(
                student_interval.to_string(),
                self.correct_interval.to_string(),
                self.answer_type,
            )

        result.preview = student_interval.to_tex()
        result.original_student_answer = student_answer
        return result
