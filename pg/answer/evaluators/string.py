"""
String answer evaluator.

Handles string comparison with various matching modes.
"""

from __future__ import annotations

import re
from typing import Any

from ..answer_hash import AnswerResult
from ..evaluator import AnswerEvaluator


class StringEvaluator(AnswerEvaluator):
    """
    Evaluator for string answers.

    Supports:
    - Exact matching
    - Case-insensitive matching
    - Regex matching
    - Trimming whitespace
    """

    answer_type = "string"

    def __init__(
        self,
        correct_answer: str,
        case_sensitive: bool = True,
        trim_whitespace: bool = True,
        regex_match: bool = False,
        **options: Any,
    ):
        """
        Initialize string evaluator.

        Args:
            correct_answer: Correct string answer
            case_sensitive: Whether matching is case-sensitive
            trim_whitespace: Trim leading/trailing whitespace
            regex_match: Treat correct_answer as regex pattern
            **options: Additional options
        """
        super().__init__(correct_answer, **options)
        self.case_sensitive = case_sensitive
        self.trim_whitespace = trim_whitespace
        self.regex_match = regex_match

        # Compile regex if needed
        if regex_match:
            flags = 0 if case_sensitive else re.IGNORECASE
            self.pattern = re.compile(correct_answer, flags)
        else:
            self.pattern = None

    def parse_student_answer(self, answer: str) -> str:
        """Parse student's string answer."""
        if self.trim_whitespace:
            answer = answer.strip()
        return answer

    def compare(self, student_value: str, correct_value: str) -> tuple[bool, float]:
        """Compare strings."""
        if self.regex_match and self.pattern:
            # Regex matching
            is_correct = bool(self.pattern.fullmatch(student_value))
        else:
            # Exact or case-insensitive matching
            if self.case_sensitive:
                is_correct = student_value == correct_value
            else:
                is_correct = student_value.lower() == correct_value.lower()

        return is_correct, 1.0 if is_correct else 0.0

    def evaluate(self, student_answer: str) -> AnswerResult:
        """Evaluate string answer."""
        student_value = self.parse_student_answer(student_answer)
        is_correct, score = self.compare(student_value, self.correct_answer)

        if is_correct:
            result = AnswerResult.correct_answer(
                student_ans=student_value,
                correct_ans=str(self.correct_answer),
                answer_type=self.answer_type,
            )
        else:
            result = AnswerResult.incorrect_answer(
                student_ans=student_value,
                correct_ans=str(self.correct_answer),
                answer_type=self.answer_type,
            )

        result.original_student_answer = student_answer
        result.metadata["case_sensitive"] = self.case_sensitive
        result.metadata["regex_match"] = self.regex_match

        return result
