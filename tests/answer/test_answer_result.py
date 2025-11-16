"""Tests for AnswerResult Pydantic model."""

import pytest
from pydantic import ValidationError

from pg.answer.answer_hash import AnswerResult


class TestAnswerResultBasicInstantiation:
    """Test basic instantiation and field validation."""

    def test_default_instantiation(self):
        """Test creating AnswerResult with default values."""
        result = AnswerResult()
        assert result.score == 0.0
        assert result.correct is False
        assert result.student_answer == ""
        assert result.student_correct_answer == ""
        assert result.answer_message == ""
        assert result.messages == []
        assert result.type == "unknown"
        assert result.error_flag is False

    def test_instantiation_with_all_fields(self):
        """Test creating AnswerResult with all fields specified."""
        result = AnswerResult(
            score=0.75,
            correct=False,
            student_answer="x+1",
            student_correct_answer="x+1",
            answer_message="Good job!",
            messages=["Check your work"],
            type="formula",
            preview="x + 1",
            error_message="",
            error_flag=False,
            ans_label="ans_1",
            metadata={"debug": "info"},
        )
        assert result.score == 0.75
        assert result.correct is False
        assert result.student_answer == "x+1"
        assert result.answer_message == "Good job!"
        assert result.type == "formula"
        assert result.messages == ["Check your work"]
        assert result.ans_label == "ans_1"
        assert result.metadata == {"debug": "info"}

    def test_score_range_validation_upper(self, assert_validation_error):
        """Test that score > 1.0 is rejected."""
        assert_validation_error(AnswerResult, {"score": 1.5}, expected_field="score")

    def test_score_range_validation_lower(self, assert_validation_error):
        """Test that score < 0.0 is rejected."""
        assert_validation_error(AnswerResult, {"score": -0.1}, expected_field="score")

    def test_score_exactly_zero(self):
        """Test that score 0.0 is valid."""
        result = AnswerResult(score=0.0)
        assert result.score == 0.0

    def test_score_exactly_one(self):
        """Test that score 1.0 is valid."""
        result = AnswerResult(score=1.0)
        assert result.score == 1.0

    def test_score_in_middle_range(self):
        """Test that partial credit scores are valid."""
        result = AnswerResult(score=0.5)
        assert result.score == 0.5


class TestAnswerResultScoreCorrectSync:
    """Test score and correct flag synchronization."""

    def test_score_1_0_sets_correct_true(self):
        """Test that score 1.0 automatically sets correct=True."""
        result = AnswerResult(score=1.0, correct=False)
        assert result.score == 1.0
        assert result.correct is True

    def test_score_0_0_sets_correct_false(self):
        """Test that score 0.0 automatically sets correct=False."""
        result = AnswerResult(score=0.0, correct=True)
        assert result.score == 0.0
        assert result.correct is False

    def test_partial_credit_keeps_custom_correct_false(self):
        """Test that partial credit (0 < score < 1) keeps correct=False."""
        result = AnswerResult(score=0.5, correct=False)
        assert result.score == 0.5
        assert result.correct is False

    def test_explicit_correct_override_for_custom_threshold(self):
        """Test that explicit correct flag is preserved for custom scoring."""
        result = AnswerResult(score=0.75, correct=True)
        assert result.score == 0.75
        assert result.correct is True


class TestAnswerResultMethods:
    """Test AnswerResult methods."""

    def test_add_message(self):
        """Test add_message() method."""
        result = AnswerResult()
        assert result.messages == []
        result.add_message("First message")
        assert result.messages == ["First message"]
        result.add_message("Second message")
        assert result.messages == ["First message", "Second message"]

    def test_add_message_deduplicates(self):
        """Test that add_message() doesn't add duplicates."""
        result = AnswerResult()
        result.add_message("Message")
        result.add_message("Message")
        assert result.messages == ["Message"]

    def test_add_message_ignores_empty(self):
        """Test that add_message() ignores empty strings."""
        result = AnswerResult()
        result.add_message("")
        result.add_message("  ")
        assert result.messages == []

    def test_set_error(self):
        """Test set_error() method."""
        result = AnswerResult(score=0.5, correct=True)
        result.set_error("Syntax error")
        assert result.error_flag is True
        assert result.error_message == "Syntax error"
        assert result.score == 0.0
        assert result.correct is False

    def test_is_correct_default_threshold(self):
        """Test is_correct() with default threshold."""
        result_correct = AnswerResult(score=1.0)
        assert result_correct.is_correct() is True

        result_incorrect = AnswerResult(score=0.9)
        assert result_incorrect.is_correct() is False

    def test_is_correct_custom_threshold(self):
        """Test is_correct() with custom threshold."""
        result = AnswerResult(score=0.75)
        assert result.is_correct(threshold=0.5) is True
        assert result.is_correct(threshold=0.75) is True
        assert result.is_correct(threshold=0.76) is False

    def test_is_partial_credit(self):
        """Test is_partial_credit() method."""
        assert AnswerResult(score=0.0).is_partial_credit() is False
        assert AnswerResult(score=0.5).is_partial_credit() is True
        assert AnswerResult(score=1.0).is_partial_credit() is False

    def test_is_blank(self):
        """Test is_blank() method."""
        assert AnswerResult(original_student_answer="").is_blank() is True
        assert AnswerResult(original_student_answer="  ").is_blank() is True
        assert AnswerResult(original_student_answer="x+1").is_blank() is False


class TestAnswerResultSerialization:
    """Test serialization and deserialization."""

    def test_to_dict(self):
        """Test to_dict() method."""
        result = AnswerResult(
            score=0.75,
            correct=False,
            student_answer="x",
            student_correct_answer="x+1",
            answer_message="Close!",
            type="formula",
            ans_label="ans_1",
        )
        data = result.to_dict()
        assert isinstance(data, dict)
        assert data["score"] == 0.75
        assert data["correct"] is False
        assert data["student_answer"] == "x"
        assert data["answer_message"] == "Close!"
        assert data["type"] == "formula"
        assert data["ans_label"] == "ans_1"

    def test_to_dict_with_math_object_references(self):
        """Test to_dict() converts MathObject references to strings."""
        class MockMathObject:
            def __str__(self):
                return "MathObject(42)"

        result = AnswerResult(
            correct_value=MockMathObject(),
            student_value=MockMathObject(),
            student_formula=MockMathObject(),
        )
        data = result.to_dict()
        assert data["correct_value"] == "MathObject(42)"
        assert data["student_value"] == "MathObject(42)"
        assert data["student_formula"] == "MathObject(42)"

    def test_from_dict(self):
        """Test from_dict() class method."""
        data = {
            "score": 0.9,
            "correct": True,
            "student_answer": "2x",
            "correct_answer": "2x",
            "answer_message": "Excellent!",
            "type": "formula",
            "ans_label": "ans_2",
        }
        result = AnswerResult.from_dict(data)
        assert result.score == 0.9
        assert result.correct is True
        assert result.student_answer == "2x"
        assert result.answer_message == "Excellent!"
        assert result.type == "formula"
        assert result.ans_label == "ans_2"

    def test_from_dict_with_missing_fields(self):
        """Test from_dict() with missing optional fields."""
        data = {"score": 0.5}
        result = AnswerResult.from_dict(data)
        assert result.score == 0.5
        assert result.correct is False
        assert result.student_answer == ""
        assert result.messages == []

    def test_round_trip_serialization(self, assert_serializable):
        """Test that serialize → deserialize preserves data."""
        original = AnswerResult(
            score=0.75,
            correct=False,
            student_answer="x+1",
            student_correct_answer="x+1",
            answer_message="Good!",
            messages=["Check work"],
            type="formula",
            metadata={"debug": "test"},
        )
        reconstructed = assert_serializable(original, AnswerResult)
        assert reconstructed.score == original.score
        assert reconstructed.student_answer == original.student_answer
        assert reconstructed.messages == original.messages
        assert reconstructed.metadata == original.metadata


class TestAnswerResultFactoryMethods:
    """Test factory methods for creating common result types."""

    def test_correct_answer_factory(self):
        """Test correct_answer() factory method."""
        result = AnswerResult.correct_answer(
            student_ans="2x",
            correct_ans="2x",
            answer_type="formula",
            message="Perfect!",
        )
        assert result.score == 1.0
        assert result.correct is True
        assert result.student_answer == "2x"
        assert result.student_correct_answer == "2x"
        assert result.type == "formula"
        assert result.answer_message == "Perfect!"

    def test_correct_answer_factory_default_message(self):
        """Test correct_answer() with default message."""
        result = AnswerResult.correct_answer("x", "x")
        assert result.score == 1.0
        assert result.answer_message == "Correct!"

    def test_incorrect_answer_factory(self):
        """Test incorrect_answer() factory method."""
        result = AnswerResult.incorrect_answer(
            student_ans="x",
            correct_ans="2x",
            answer_type="formula",
            message="Try again!",
        )
        assert result.score == 0.0
        assert result.correct is False
        assert result.student_answer == "x"
        assert result.student_correct_answer == "2x"
        assert result.answer_message == "Try again!"

    def test_incorrect_answer_factory_default_message(self):
        """Test incorrect_answer() with default message."""
        result = AnswerResult.incorrect_answer("x", "2x")
        assert result.score == 0.0
        assert result.answer_message == "Incorrect."

    def test_error_answer_factory(self):
        """Test error_answer() factory method."""
        result = AnswerResult.error_answer(
            student_ans="x+",
            error="Syntax error: incomplete expression",
            answer_type="formula",
        )
        assert result.score == 0.0
        assert result.correct is False
        assert result.error_flag is True
        assert result.error_message == "Syntax error: incomplete expression"
        assert result.original_student_answer == "x+"

    def test_partial_credit_answer_factory(self):
        """Test partial_credit_answer() factory method."""
        result = AnswerResult.partial_credit_answer(
            score=0.75,
            student_ans="x + 1",
            correct_ans="x+1",
            answer_type="formula",
            message="Mostly right",
        )
        assert result.score == 0.75
        assert result.correct is False
        assert result.student_answer == "x + 1"
        assert result.answer_message == "Mostly right"

    def test_partial_credit_answer_factory_default_message(self):
        """Test partial_credit_answer() with default message."""
        result = AnswerResult.partial_credit_answer(0.5, "x", "x+1")
        assert result.score == 0.5
        assert result.answer_message == "Partially correct (50%)."


class TestAnswerResultEdgeCases:
    """Test edge cases and special scenarios."""

    def test_metadata_is_mutable_dict(self):
        """Test that metadata dict can be modified."""
        result = AnswerResult()
        assert result.metadata == {}
        result.metadata["key"] = "value"
        assert result.metadata["key"] == "value"

    def test_messages_list_is_mutable(self):
        """Test that messages list can be modified."""
        result = AnswerResult()
        result.messages.append("Added after init")
        assert "Added after init" in result.messages

    def test_score_type_coercion(self):
        """Test that integer score is accepted and converted to float."""
        result = AnswerResult(score=1)
        assert result.score == 1.0
        assert isinstance(result.score, float)

    def test_correct_flag_type_validation(self, assert_validation_error):
        """Test that correct must be boolean."""
        assert_validation_error(AnswerResult, {"correct": "yes"}, expected_field="correct")

    def test_type_error_flag_field(self):
        """Test typeError field (Perl parity)."""
        result = AnswerResult(typeError=True)
        assert result.typeError is True

    def test_preview_field_can_hold_latex(self):
        """Test preview field can store LaTeX/HTML."""
        result = AnswerResult(preview="$x + 1$")
        assert result.preview == "$x + 1$"

    def test_math_object_references_are_optional(self):
        """Test that math object references default to None."""
        result = AnswerResult()
        assert result.correct_value is None
        assert result.student_value is None
        assert result.student_formula is None

    def test_multiple_add_message_calls(self):
        """Test multiple message additions."""
        result = AnswerResult()
        result.add_message("Step 1")
        result.add_message("Step 2")
        result.add_message("Step 3")
        assert len(result.messages) == 3

    def test_set_error_clears_score_and_correct(self):
        """Test that set_error() properly resets score and correct."""
        result = AnswerResult(score=0.9, correct=True, answer_message="Great!")
        result.set_error("Something went wrong")
        assert result.score == 0.0
        assert result.correct is False
        assert result.error_flag is True


class TestAnswerResultIntegration:
    """Integration tests combining multiple features."""

    def test_workflow_correct_answer(self):
        """Test typical workflow for correct answer."""
        result = AnswerResult.correct_answer("42", "42", "numeric")
        assert result.is_correct()
        assert not result.is_partial_credit()
        assert not result.is_blank()
        result.add_message("Bonus: great speed!")
        assert len(result.messages) == 1

    def test_workflow_partial_credit(self):
        """Test typical workflow for partial credit."""
        result = AnswerResult.partial_credit_answer(
            0.5, "x", "2x", "formula", "Sign error"
        )
        assert result.is_partial_credit()
        assert not result.is_correct()
        data = result.to_dict()
        assert data["score"] == 0.5

    def test_workflow_error_recovery(self):
        """Test workflow with error handling."""
        result = AnswerResult(original_student_answer="x+")
        assert result.is_blank() is False
        result.set_error("Syntax error")
        assert result.error_flag is True
        reconstructed = AnswerResult.from_dict(result.to_dict())
        assert reconstructed.error_flag is True
