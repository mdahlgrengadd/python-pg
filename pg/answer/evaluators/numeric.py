"""
Numeric answer evaluator.

Handles evaluation of numeric answers (Real, Complex, Infinity) with
fuzzy comparison tolerances.

Reference: lib/Value/Real.pm answer checking (lines 200-250)
"""

from __future__ import annotations

from typing import Any

from pg.math import Complex, Infinity, MathValue, Real, ToleranceMode
from pg.parser import Parser

from ..answer_hash import AnswerResult
from ..evaluator import AnswerEvaluator


class NumericEvaluator(AnswerEvaluator):
    """
    Evaluator for numeric answers (Real, Complex, Infinity).

    Supports:
    - Real numbers with fuzzy comparison
    - Complex numbers
    - Infinity (+inf, -inf)
    - Multiple tolerance modes (relative, absolute, sigfigs)
    - Expression evaluation (e.g., "2*pi", "sqrt(2)")
    """

    answer_type = "numeric"

    def __init__(
        self,
        correct_answer: MathValue | float | complex | str,
        tolerance: float = 0.001,
        tolerance_mode: str = ToleranceMode.RELATIVE,
        allow_expressions: bool = True,
        **options: Any,
    ):
        """
        Initialize numeric evaluator.

        Args:
            correct_answer: Correct numeric answer
            tolerance: Tolerance for fuzzy comparison (default 0.001 = 0.1%)
            tolerance_mode: "relative", "absolute", or "sigfigs"
            allow_expressions: Allow student to enter expressions like "2*pi"
            **options: Additional options
        """
        # Set instance variables FIRST before parsing
        self.allow_expressions = allow_expressions
        self.parser = Parser() if allow_expressions else None

        # Convert correct answer to MathValue
        if isinstance(correct_answer, MathValue):
            self.correct_value = correct_answer
        elif isinstance(correct_answer, str):
            # Parse string expression
            self.correct_value = self._parse_numeric(correct_answer)
        elif isinstance(correct_answer, complex):
            self.correct_value = Complex(correct_answer.real, correct_answer.imag)
        elif isinstance(correct_answer, (int, float)):
            self.correct_value = Real(float(correct_answer))
        else:
            raise TypeError(f"Invalid correct answer type: {type(correct_answer)}")

        super().__init__(
            correct_answer=self.correct_value,
            tolerance=tolerance,
            tolerance_mode=tolerance_mode,
            **options,
        )

    def _parse_numeric(self, answer: str) -> MathValue:
        """
        Parse numeric answer from string.

        Args:
            answer: String input (number or expression)

        Returns:
            MathValue (Real, Complex, or Infinity)

        Raises:
            ValueError: If answer cannot be parsed
        """
        answer = answer.strip()

        if not answer:
            raise ValueError("Empty answer")

        # Handle special cases
        if answer.lower() in ("inf", "infinity", "+inf", "+infinity"):
            return Infinity(1)
        elif answer.lower() in ("-inf", "-infinity"):
            return Infinity(-1)
        elif answer.lower() in ("nan", "undefined"):
            return Infinity(0)  # Undefined

        # Try to parse as simple number first
        try:
            # Try as integer
            if "." not in answer and "e" not in answer.lower():
                return Real(float(int(answer)))
            else:
                # Try as float
                return Real(float(answer))
        except ValueError:
            pass

        # Try as complex number
        try:
            c = complex(answer)
            return Complex(c.real, c.imag)
        except ValueError:
            pass

        # Try parsing as expression (if allowed)
        if self.allow_expressions and self.parser:
            try:
                from pg.parser.visitors import EvalVisitor

                ast = self.parser.parse(answer)
                visitor = EvalVisitor()
                result = ast.accept(visitor)

                # Convert to MathValue
                if isinstance(result, complex):
                    return Complex(result.real, result.imag)
                elif isinstance(result, float):
                    if result == float("inf"):
                        return Infinity(1)
                    elif result == float("-inf"):
                        return Infinity(-1)
                    else:
                        return Real(result)
                else:
                    return MathValue.from_python(result)

            except Exception as e:
                raise ValueError(f"Could not parse expression: {e}") from e

        raise ValueError(f"Could not parse as numeric value: {answer}")

    def parse_student_answer(self, answer: str) -> MathValue:
        """Parse student's numeric answer."""
        return self._parse_numeric(answer)

    def compare(
        self, student_value: MathValue, correct_value: MathValue
    ) -> tuple[bool, float]:
        """
        Compare numeric values with fuzzy tolerance.

        Args:
            student_value: Student's parsed answer
            correct_value: Correct answer

        Returns:
            Tuple of (is_correct, score)
        """
        # Use MathValue's fuzzy comparison
        is_correct = student_value.compare(
            correct_value,
            tolerance=self.tolerance,
            mode=self.tolerance_mode,
        )

        return is_correct, 1.0 if is_correct else 0.0

    def evaluate(self, student_answer: str) -> AnswerResult:
        """
        Evaluate numeric answer.

        Args:
            student_answer: Student's input string

        Returns:
            AnswerResult with score and feedback
        """
        # Handle blank answer
        if not student_answer.strip():
            return AnswerResult.error_answer(
                student_answer,
                "Answer cannot be blank",
                answer_type=self.answer_type,
            )

        # Try to parse student answer
        try:
            student_value = self.parse_student_answer(student_answer)
        except ValueError as e:
            return AnswerResult.error_answer(
                student_answer,
                f"Could not parse answer: {e}",
                answer_type=self.answer_type,
            )

        # Compare with correct answer
        is_correct, score = self.compare(student_value, self.correct_value)

        # Build result
        if is_correct:
            result = AnswerResult.correct_answer(
                student_ans=student_value.to_string(),
                correct_ans=self.correct_value.to_string(),
                answer_type=self.answer_type,
                message="Correct!",
            )
        else:
            result = AnswerResult.incorrect_answer(
                student_ans=student_value.to_string(),
                correct_ans=self.correct_value.to_string(),
                answer_type=self.answer_type,
                message="Incorrect.",
            )

        # Set preview (LaTeX)
        result.preview = student_value.to_tex()
        result.original_student_answer = student_answer

        # Add metadata
        result.metadata["tolerance"] = self.tolerance
        result.metadata["tolerance_mode"] = self.tolerance_mode
        result.metadata["parsed_value"] = student_value.to_python()

        return result

    def get_preview(self, student_value: MathValue) -> str:
        """Generate LaTeX preview of numeric answer."""
        return student_value.to_tex()

    def get_correct_answer_display(self) -> str:
        """Get display string for correct answer."""
        return self.correct_value.to_string()
