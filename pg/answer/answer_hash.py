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

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnswerResult:
    """
    Result of answer evaluation.

    This is the Python equivalent of Perl's AnswerHash, containing all
    information about how a student's answer was evaluated.

    Attributes:
        score: Correctness score (0.0 = wrong, 1.0 = correct, partial credit in between)
        correct: Boolean indicating if answer is considered correct
        student_answer: Student's answer (original input)
        correct_answer: The correct answer (for display)
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

    # Core fields (always present)
    score: float = 0.0  # 0.0 to 1.0
    correct: bool = False  # True if score >= 1.0 (or custom threshold)

    # Student/correct answer fields
    student_answer: str = ""  # Parsed/normalized student answer
    correct_answer: str = ""  # Correct answer for display
    original_student_answer: str = ""  # Raw student input

    # Feedback messages
    answer_message: str = ""  # Primary message
    messages: list[str] = field(default_factory=list)  # Additional messages

    # Answer type and preview
    type: str = "unknown"  # numeric, formula, string, interval, vector, etc.
    preview: str = ""  # LaTeX or HTML preview

    # Error handling
    error_message: str = ""  # Error description
    error_flag: bool = False  # True if evaluation failed
    typeError: bool = False  # True if type mismatch (Perl parity)

    # Answer blank identification
    ans_label: str = ""  # e.g., "ans_1", "ans_2"

    # MathObject references (for parity with Perl AnswerHash)
    correct_value: Any = None  # MathObject reference for correct answer
    student_value: Any = None  # MathObject reference for parsed student answer
    student_formula: Any = None  # Formula reference for student answer (if applicable)

    # Metadata for debugging/extensions
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize fields after initialization."""
        # Ensure score is in valid range
        self.score = max(0.0, min(1.0, self.score))

        # Sync correct flag with score (1.0 = correct)
        if self.score >= 1.0:
            self.correct = True
        elif self.score <= 0.0:
            self.correct = False
        # Otherwise keep custom correct value (for custom thresholds)

    def add_message(self, message: str) -> None:
        """Add a feedback message."""
        if message and message not in self.messages:
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
        return not self.original_student_answer.strip()

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
            "correct_answer": self.correct_answer,
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
        return cls(
            score=data.get("score", 0.0),
            correct=data.get("correct", False),
            student_answer=data.get("student_answer", ""),
            correct_answer=data.get("correct_answer", ""),
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
    def correct_answer(
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
            correct_answer=correct_ans,
            type=answer_type,
            answer_message=message or "Correct!",
        )

    @classmethod
    def incorrect_answer(
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
            correct_answer=correct_ans,
            type=answer_type,
            answer_message=message or "Incorrect.",
        )

    @classmethod
    def error_answer(
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
    def partial_credit_answer(
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
            correct=False,  # Not fully correct
            student_answer=student_ans,
            correct_answer=correct_ans,
            type=answer_type,
            answer_message=message or f"Partially correct ({score * 100:.0f}%).",
        )
