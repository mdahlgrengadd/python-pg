"""
Formula answer evaluator.

Handles evaluation of formula answers with symbolic comparison
and test point evaluation.

Reference: lib/Value/Formula.pm answer checking (lines 800-900)
"""

from __future__ import annotations

import random
from typing import Any

from pg.math import Formula, MathValue, Real, ToleranceMode
from pg.parser import Parser

from ..answer_hash import AnswerResult
from ..evaluator import AnswerEvaluator


class FormulaEvaluator(AnswerEvaluator):
    """
    Evaluator for formula answers.

    Supports:
    - Symbolic comparison (using SymPy)
    - Test point evaluation (fallback if symbolic fails)
    - Variable validation
    - Multiple tolerance modes
    """

    answer_type = "formula"

    def __init__(
        self,
        correct_answer: Formula | str,
        tolerance: float = 0.001,
        tolerance_mode: str = ToleranceMode.RELATIVE,
        variables: list[str] | None = None,
        test_points: int = 5,
        test_at_zero: bool = True,
        up_to_additive_constant: bool = False,
        limits: dict[str, tuple[float, float]] | None = None,
        check_undefined_points: bool = False,
        **options: Any,
    ):
        """
        Initialize formula evaluator.

        Args:
            correct_answer: Correct formula
            tolerance: Tolerance for numeric comparison at test points
            tolerance_mode: "relative", "absolute", or "sigfigs"
            variables: List of allowed variables (None = infer from formula)
            test_points: Number of random test points for evaluation
            test_at_zero: Always test at x=0, y=0, etc.
            **options: Additional options
        """
        # Convert correct answer to Formula
        if isinstance(correct_answer, Formula):
            self.correct_formula = correct_answer
        elif isinstance(correct_answer, str):
            self.correct_formula = Formula(
                correct_answer, variables=variables or []
            )
        else:
            raise TypeError(f"Invalid correct answer type: {type(correct_answer)}")

        super().__init__(
            correct_answer=self.correct_formula,
            tolerance=tolerance,
            tolerance_mode=tolerance_mode,
            **options,
        )

        self.variables = variables or self.correct_formula.variables
        self.test_points = test_points
        self.test_at_zero = test_at_zero
        self.up_to_additive_constant = up_to_additive_constant
        self.limits = limits or {}
        self.check_undefined_points = check_undefined_points
        self.parser = Parser()

    def parse_student_answer(self, answer: str) -> Formula:
        """
        Parse student's formula answer.

        Args:
            answer: String expression

        Returns:
            Formula object

        Raises:
            ValueError: If answer cannot be parsed
        """
        answer = answer.strip()

        if not answer:
            raise ValueError("Empty answer")

        try:
            return Formula(answer, variables=self.variables)
        except Exception as e:
            raise ValueError(f"Could not parse formula: {e}") from e

    def compare(
        self, student_formula: Formula, correct_formula: Formula
    ) -> tuple[bool, float]:
        """
        Compare formulas using symbolic and test-point methods.

        Args:
            student_formula: Student's formula
            correct_formula: Correct formula

        Returns:
            Tuple of (is_correct, score)
        """
        # Apply evaluator limits and undefined options to both formulas
        try:
            student_formula._limits = self.limits or getattr(student_formula, '_limits', {})
            correct_formula._limits = self.limits or getattr(correct_formula, '_limits', {})
            student_formula.check_undefined_points = self.check_undefined_points
            correct_formula.check_undefined_points = self.check_undefined_points
        except Exception:
            pass

        # Additive constant parity (antiderivative-style)
        if self.up_to_additive_constant:
            is_correct = self._check_up_to_additive_constant(student_formula, correct_formula)
        else:
            # Use Formula's built-in comparison (symbolic + test points)
            is_correct = student_formula.compare(
                correct_formula,
                tolerance=self.tolerance,
                mode=self.tolerance_mode,
            )

        return is_correct, 1.0 if is_correct else 0.0

    def _test_at_points(
        self, student_formula: Formula, correct_formula: Formula
    ) -> tuple[bool, list[dict[str, Any]]]:
        """
        Test formulas at specific points for detailed feedback.

        Args:
            student_formula: Student's formula
            correct_formula: Correct formula

        Returns:
            Tuple of (all_match, test_results)
            - all_match: True if formulas match at all test points
            - test_results: List of test point results with details
        """
        test_results = []
        all_match = True

        # Generate test points
        random.seed(12345)  # Deterministic

        # Test at zero if requested
        if self.test_at_zero and self.variables:
            zero_bindings = {var: 0.0 for var in self.variables}
            try:
                student_val = student_formula.eval(**zero_bindings)
                correct_val = correct_formula.eval(**zero_bindings)

                matches = student_val.compare(
                    correct_val,
                    tolerance=self.tolerance,
                    mode=self.tolerance_mode,
                )

                test_results.append({
                    "bindings": zero_bindings,
                    "student_value": student_val.to_python(),
                    "correct_value": correct_val.to_python(),
                    "matches": matches,
                })

                if not matches:
                    all_match = False

            except Exception:
                # Skip if evaluation fails at zero
                pass

        # Test at random points
        for _ in range(self.test_points):
            # Generate random bindings
            bindings = {
                var: random.uniform(-10, 10) for var in self.variables
            }

            try:
                student_val = student_formula.eval(**bindings)
                correct_val = correct_formula.eval(**bindings)

                matches = student_val.compare(
                    correct_val,
                    tolerance=self.tolerance,
                    mode=self.tolerance_mode,
                )

                test_results.append({
                    "bindings": bindings,
                    "student_value": student_val.to_python(),
                    "correct_value": correct_val.to_python(),
                    "matches": matches,
                })

                if not matches:
                    all_match = False

            except (ValueError, ZeroDivisionError):
                # Skip test points that cause errors
                continue

        return all_match, test_results

    def _check_up_to_additive_constant(self, student_formula: Formula, correct_formula: Formula) -> bool:
        """Check if student = correct + C for some constant C by sampling.

        Strategy: evaluate (student - correct) at multiple points and verify
        the differences are approximately equal.
        """
        import random
        random.seed(12345)

        # Fast path: if both have no variables, any finite constant difference is OK
        if not student_formula.variables and not correct_formula.variables:
            try:
                diff = student_formula - correct_formula
                # If subtraction succeeds, treat as constant difference
                return True
            except Exception:
                return False

        diffs: list[float] = []
        vars_ = self.variables or student_formula.variables or correct_formula.variables
        if not vars_:
            # No variables known; fallback to standard compare
            return student_formula.compare(correct_formula, tolerance=self.tolerance, mode=self.tolerance_mode)

        for _ in range(self.test_points):
            bindings = {v: random.uniform(-10, 10) for v in vars_}
            try:
                s = student_formula.eval(**bindings).to_python()
                c = correct_formula.eval(**bindings).to_python()
                diffs.append(float(s - c))
            except Exception:
                continue

        if len(diffs) < 3:
            return False

        baseline = diffs[0]
        return all(abs(d - baseline) <= self.tolerance for d in diffs)

    def evaluate(self, student_answer: str) -> AnswerResult:
        """
        Evaluate formula answer.

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
            student_formula = self.parse_student_answer(student_answer)
        except ValueError as e:
            return AnswerResult.error_answer(
                student_answer,
                f"Could not parse formula: {e}",
                answer_type=self.answer_type,
            )

        # Compare formulas
        is_correct, score = self.compare(student_formula, self.correct_formula)

        # Get detailed test results for feedback
        _, test_results = self._test_at_points(
            student_formula, self.correct_formula
        )

        # Build result
        if is_correct:
            result = AnswerResult.correct_answer(
                student_ans=student_formula.to_string(),
                correct_ans=self.correct_formula.to_string(),
                answer_type=self.answer_type,
                message="Correct!",
            )
        else:
            result = AnswerResult.incorrect_answer(
                student_ans=student_formula.to_string(),
                correct_ans=self.correct_formula.to_string(),
                answer_type=self.answer_type,
                message="Incorrect.",
            )

            # Add feedback about first failing test point
            for test in test_results:
                if not test["matches"]:
                    bindings_str = ", ".join(
                        f"{var}={val:.2f}" for var, val in test["bindings"].items()
                    )
                    result.add_message(
                        f"Your answer differs from the correct answer at {bindings_str}"
                    )
                    break

        # Set preview (LaTeX)
        result.preview = student_formula.to_tex()
        result.original_student_answer = student_answer

        # Add metadata
        result.metadata["tolerance"] = self.tolerance
        result.metadata["tolerance_mode"] = self.tolerance_mode
        result.metadata["variables"] = self.variables
        result.metadata["test_results"] = test_results

        return result

    def get_preview(self, student_formula: Formula) -> str:
        """Generate LaTeX preview of formula."""
        return student_formula.to_tex()

    def get_correct_answer_display(self) -> str:
        """Get display string for correct answer."""
        return self.correct_formula.to_string()
