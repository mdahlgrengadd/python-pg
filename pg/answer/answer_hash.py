"""
Answer result data structure (AnswerHash equivalent).

This module provides the AnswerResult class which encapsulates the result
of answer evaluation, including:
- Correctness score (0.0 to 1.0)
- Student/correct answers
- Messages and feedback
- Metadata for debugging

Reference: lib/AnswerHash.pm (lines 1-300) in legacy Perl codebase
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, StrictBool


class AnswerResult(BaseModel):
    """
    Result of answer evaluation.

    This is the Python equivalent of Perl's AnswerHash, containing all
    information about how a student's answer was evaluated.

    Attributes:
        score: Correctness score (0.0 = wrong, 1.0 = correct, partial credit in between)
        correct: Boolean indicating if answer is considered correct
        student_answer: Student's answer (original input)
        student_correct_answer: The correct answer (for display)
        answer_message: Primary feedback message shown to student
        messages: List of additional feedback messages
        type: Type of answer (numeric, formula, string, etc.)
        preview: LaTeX/HTML preview of student's answer
        original_student_answer: Unparsed student input
        error_message: Error message if answer couldn't be evaluated
        error_flag: Boolean indicating evaluation error
        ans_label: Answer blank label (ans_1, ans_2, etc.)
        metadata: Additional metadata for debugging/analysis
    """

    model_config = ConfigDict(validate_assignment=True)

    # Core fields (always present)
    score: float = 0.0
    correct: StrictBool = False

    # Student/correct answer fields
    student_answer: str = ""
    student_correct_answer: str = ""
    original_student_answer: str = ""

    # Feedback messages
    answer_message: str = ""
    messages: list[str] = []

    # Answer type and preview
    type: str = "unknown"
    preview: str = ""

    # Error handling
    error_message: str = ""
    error_flag: bool = False
    typeError: bool = False

    # Answer blank identification
    ans_label: str = ""

    # MathObject references (for parity with Perl AnswerHash)
    correct_value: Any = None
    student_value: Any = None
    student_formula: Any = None

    # Metadata for debugging/extensions
    metadata: dict[str, Any] = {}

    def model_post_init(self, __context: Any) -> None:
        """Validate and normalize fields after initialization."""
        # Ensure score is in valid range
        self.score = max(0.0, min(1.0, self.score))

        # Sync correct flag with score (1.0 = correct)
        if self.score >= 1.0:
            self.correct = True
        elif self.score <= 0.0:
            self.correct = False

        # Backfill original answer if we only captured a parsed version
        if not self.original_student_answer and self.student_answer:
            self.original_student_answer = self.student_answer

        if self.score >= 1.0:
            self.correct = True
        elif self.score <= 0.0:
            self.correct = False

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Validate score is in valid range."""
        if not isinstance(v, (int, float)):
            raise ValueError("score must be numeric")
        if v < 0.0 or v > 1.0:
            raise ValueError("score must be between 0.0 and 1.0")
        return float(v)

    @field_validator("messages", mode="before")
    @classmethod
    def validate_messages(cls, v: Any) -> list[str]:
        """Ensure messages is a list."""
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("messages must be a list")
        return v

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, v: Any) -> dict[str, Any]:
        """Ensure metadata is a dict."""
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("metadata must be a dict")
        return v

    def add_message(self, message: str) -> None:
        """Add a feedback message."""
        if message and message.strip() and message not in self.messages:
            self.messages.append(message)

    def set_error(self, error: str) -> None:
        """Mark answer as having an error."""
        self.error_flag = True
        self.error_message = error
        self.score = 0.0
        self.correct = False

    def is_correct(self, threshold: float = 1.0) -> bool:
        """
        Check if answer is correct based on score threshold.

        Args:
            threshold: Minimum score to be considered correct (default 1.0)

        Returns:
            True if score >= threshold
        """
        return self.score >= threshold

    def is_partial_credit(self) -> bool:
        """Check if answer received partial credit."""
        return 0.0 < self.score < 1.0

    def is_blank(self) -> bool:
        """Check if student answer is blank."""
        return not (
            self.original_student_answer.strip() or self.student_answer.strip()
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON/API responses
        """
        result = {
            "score": self.score,
            "correct": self.correct,
            "student_answer": self.student_answer,
            "student_correct_answer": self.student_correct_answer,
            "original_student_answer": self.original_student_answer,
            "answer_message": self.answer_message,
            "messages": self.messages,
            "type": self.type,
            "preview": self.preview,
            "error_message": self.error_message,
            "error_flag": self.error_flag,
            "typeError": self.typeError,
            "ans_label": self.ans_label,
            "metadata": self.metadata,
        }

        # Add MathObject references as strings (for serialization)
        if self.correct_value is not None:
            result["correct_value"] = str(self.correct_value)
        if self.student_value is not None:
            result["student_value"] = str(self.student_value)
        if self.student_formula is not None:
            result["student_formula"] = str(self.student_formula)

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnswerResult:
        """
        Create AnswerResult from dictionary.

        Args:
            data: Dictionary with answer result fields

        Returns:
            AnswerResult instance
        """
        # Handle legacy field name
        correct_answer = data.get("student_correct_answer") or data.get("correct_answer", "")
        return cls(
            score=data.get("score", 0.0),
            correct=data.get("correct", False),
            student_answer=data.get("student_answer", ""),
            student_correct_answer=correct_answer,
            original_student_answer=data.get("original_student_answer", ""),
            answer_message=data.get("answer_message", ""),
            messages=data.get("messages", []),
            type=data.get("type", "unknown"),
            preview=data.get("preview", ""),
            error_message=data.get("error_message", ""),
            error_flag=data.get("error_flag", False),
            ans_label=data.get("ans_label", ""),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def answer_correct(
        cls,
        student_ans: str,
        correct_ans: str,
        answer_type: str = "unknown",
        message: str = "",
    ) -> AnswerResult:
        """
        Create a correct answer result (convenience factory).

        Args:
            student_ans: Student's answer
            correct_ans: Correct answer
            answer_type: Type of answer
            message: Optional feedback message

        Returns:
            AnswerResult with score=1.0, correct=True
        """
        return cls(
            score=1.0,
            correct=True,
            student_answer=student_ans,
            student_correct_answer=correct_ans,
            type=answer_type,
            answer_message=message or "Correct!",
        )

    @classmethod
    def answer_incorrect(
        cls,
        student_ans: str,
        correct_ans: str,
        answer_type: str = "unknown",
        message: str = "",
    ) -> AnswerResult:
        """
        Create an incorrect answer result (convenience factory).

        Args:
            student_ans: Student's answer
            correct_ans: Correct answer
            answer_type: Type of answer
            message: Optional feedback message

        Returns:
            AnswerResult with score=0.0, correct=False
        """
        return cls(
            score=0.0,
            correct=False,
            student_answer=student_ans,
            student_correct_answer=correct_ans,
            type=answer_type,
            answer_message=message or "Incorrect.",
        )

    @classmethod
    def answer_error(
        cls,
        student_ans: str,
        error: str,
        answer_type: str = "unknown",
    ) -> AnswerResult:
        """
        Create an error answer result (convenience factory).

        Args:
            student_ans: Student's answer (raw input)
            error: Error message
            answer_type: Type of answer

        Returns:
            AnswerResult with error_flag=True, score=0.0
        """
        result = cls(
            score=0.0,
            correct=False,
            original_student_answer=student_ans,
            student_answer=student_ans,
            type=answer_type,
        )
        result.set_error(error)
        return result

    @classmethod
    def answer_partial(
        cls,
        score: float,
        student_ans: str,
        correct_ans: str,
        answer_type: str = "unknown",
        message: str = "",
    ) -> AnswerResult:
        """
        Create a partial credit answer result (convenience factory).

        Args:
            score: Partial credit score (0.0 to 1.0)
            student_ans: Student's answer
            correct_ans: Correct answer
            answer_type: Type of answer
            message: Optional feedback message

        Returns:
            AnswerResult with partial credit score
        """
        return cls(
            score=score,
            correct=False,
            student_answer=student_ans,
            student_correct_answer=correct_ans,
            type=answer_type,
            answer_message=message or f"Partially correct ({score * 100:.0f}%).",
        )

    # Legacy method names for backward compatibility
    @classmethod
    def correct_answer(
        cls,
        student_ans: str,
        correct_ans: str,
        answer_type: str = "unknown",
        message: str = "",
    ) -> AnswerResult:
        """Legacy alias for answer_correct()."""
        return cls.answer_correct(student_ans, correct_ans, answer_type, message)

    @classmethod
    def incorrect_answer(
        cls,
        student_ans: str,
        correct_ans: str,
        answer_type: str = "unknown",
        message: str = "",
    ) -> AnswerResult:
        """Legacy alias for answer_incorrect()."""
        return cls.answer_incorrect(student_ans, correct_ans, answer_type, message)

    @classmethod
    def error_answer(
        cls,
        student_ans: str,
        error: str,
        answer_type: str = "unknown",
    ) -> AnswerResult:
        """Legacy alias for answer_error()."""
        return cls.answer_error(student_ans, error, answer_type)

    @classmethod
    def partial_credit_answer(
        cls,
        score: float,
        student_ans: str,
        correct_ans: str,
        answer_type: str = "unknown",
        message: str = "",
    ) -> AnswerResult:
        """Legacy alias for answer_partial()."""
        return cls.answer_partial(score, student_ans, correct_ans, answer_type, message)
