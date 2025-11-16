"""
Problem Grading System for PG Translator.

Provides grader functions and answer processing with checkbox/radio support.
Reference: Translator.pm:848-959, 986-1142
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from pg.answer import AnswerResult


class ProblemGrader(Protocol):
    """Protocol for problem graders."""

    def __call__(
        self,
        answers: dict[str, AnswerResult],
        problem_state: dict[str, Any],
        **options: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Grade problem.

        Args:
            answers: Answer results by name
            problem_state: Current problem state
            **options: Grading options

        Returns:
            (problem_result, updated_state)
        """
        ...


def process_checkbox_radio_input(response: Any) -> Any:
    """
    Process checkbox/radio button input.

    Reference: Translator.pm:908-912

    Checkboxes/radio buttons come in as:
        [(value, "CHECKED"), (value, ""), ...]

    Args:
        response: Raw response value

    Returns:
        Processed response (single value or list)
    """
    if isinstance(response, list):
        # Check if it's checkbox/radio format
        if all(isinstance(item, (tuple, list)) and len(item) == 2 for item in response):
            # Extract checked values
            checked = [val for val, status in response if status == "CHECKED"]

            if len(checked) == 0:
                # No selection
                return ""
            elif len(checked) == 1:
                # Single selection (radio button)
                return checked[0]
            else:
                # Multiple selections (checkbox)
                return checked

    return response


def std_problem_grader(
    answers: dict[str, AnswerResult],
    problem_state: dict[str, Any],
    **options: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Standard all-or-nothing grader.

    Equivalent to Translator.pm:1014-1068

    Returns score of 1 if all answers correct, 0 otherwise.

    Args:
        answers: Evaluated answer results
        problem_state: Current problem state
        **options: Grading options (answers_submitted, etc.)

    Returns:
        (problem_result, updated_state)
    """
    # Copy state (don't modify input)
    state = dict(problem_state)

    # Initialize result
    result = {
        "score": 0,
        "errors": "",
        "type": "std_problem_grader",
        "msg": ""
    }

    # Check if we have answers
    if not answers:
        result["msg"] = "This problem did not ask any questions."
        return (result, state)

    # Multi-answer message
    if len(answers) > 1:
        result["msg"] = "In order to get credit for this problem all answers must be correct."

    # Only grade if answers submitted
    if not options.get("answers_submitted"):
        return (result, state)

    # Check all answers
    all_correct = True
    for ans_name, answer in answers.items():
        score = answer.score if isinstance(answer, AnswerResult) else answer.get("score", 0)

        if score != 1:
            all_correct = False
            break

    # Set score
    result["score"] = 1 if all_correct else 0

    # Update state
    state.setdefault("recorded_score", 0)

    if all_correct or state["recorded_score"] == 1:
        state["recorded_score"] = 1
    else:
        state["recorded_score"] = 0

    # Update attempt counters
    state.setdefault("num_of_correct_ans", 0)
    state.setdefault("num_of_incorrect_ans", 0)

    if all_correct:
        state["num_of_correct_ans"] += 1
    else:
        state["num_of_incorrect_ans"] += 1

    return (result, state)


def avg_problem_grader(
    answers: dict[str, AnswerResult],
    problem_state: dict[str, Any],
    **options: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Average (partial credit) grader.

    Equivalent to Translator.pm:1075-1142

    Returns weighted average of answer scores.

    Args:
        answers: Evaluated answer results
        problem_state: Current problem state
        **options: Grading options

    Returns:
        (problem_result, updated_state)
    """
    state = dict(problem_state)

    result = {
        "score": 0,
        "errors": "",
        "type": "avg_problem_grader",
        "msg": ""
    }

    # Multi-answer message
    if len(answers) > 1:
        result["msg"] = "You can earn partial credit on this problem."

    # Only grade if submitted
    if not options.get("answers_submitted"):
        return (result, state)

    # Calculate credit for each answer
    credit: dict[str, float] = {}
    for ans_name, answer in answers.items():
        if isinstance(answer, AnswerResult):
            credit[ans_name] = answer.score
        else:
            credit[ans_name] = answer.get("score", 0)

    # Handle optional answers (credit from other answers)
    for ans_name, answer in answers.items():
        current_credit = credit[ans_name]

        if current_credit == 1:
            # Check if this answer gives credit to others
            credit_from = None

            if isinstance(answer, AnswerResult):
                credit_from = getattr(answer, "credit_from", None)
            else:
                credit_from = answer.get("credit_from")

            if credit_from:
                # Give credit to related optional answers if blank
                credit_list = credit_from if isinstance(credit_from, list) else [credit_from]

                for credit_name in credit_list:
                    if credit_name in answers:
                        target_answer = answers[credit_name]

                        # Check if blank
                        student_ans = None
                        if isinstance(target_answer, AnswerResult):
                            student_ans = target_answer.student_answer
                        else:
                            student_ans = target_answer.get("student_answer")

                        if not student_ans or str(student_ans).strip() == "":
                            # Mark as correct
                            credit[credit_name] = 1

                            # Add message
                            msg = "This answer was marked correct because the primary answer is correct."
                            if isinstance(target_answer, AnswerResult):
                                target_answer.ans_message = msg
                            else:
                                target_answer["ans_message"] = msg

    # Calculate weighted average
    total_weight = 0.0
    total_score = 0.0

    for ans_name, answer in answers.items():
        # Get weight
        weight = 1.0
        if isinstance(answer, AnswerResult):
            weight = getattr(answer, "weight", 1.0)
        else:
            weight = answer.get("weight", 1.0)

        total_weight += weight
        total_score += weight * credit[ans_name]

    result["score"] = total_score / total_weight if total_weight > 0 else 0

    # Update state
    state.setdefault("num_of_correct_ans", 0)
    state.setdefault("num_of_incorrect_ans", 0)

    if total_score == total_weight:
        state["num_of_correct_ans"] += 1
    elif total_score < total_weight:
        state["num_of_incorrect_ans"] += 1

    state.setdefault("recorded_score", 0)

    # Increase recorded score if the current score is greater
    state["recorded_score"] = max(
        state["recorded_score"],
        result["score"]
    )

    # Warning if score > total
    if total_score > total_weight:
        import warnings
        warnings.warn(
            f"Error in grading this problem: The score {total_score} is larger than the total {total_weight}."
        )

    return (result, state)


def stringify_answers(answer_results: dict[str, AnswerResult]) -> None:
    """
    Convert all MathObject answers to strings.

    Equivalent to Translator.pm:961-977

    Ensures answer hashes contain only primitive types (str, int, float)
    for serialization.

    Args:
        answer_results: Answer results to stringify
    """
    for ans_name, result in answer_results.items():
        if hasattr(result, "stringify_hash"):
            # AnswerHash with stringify method
            result.stringify_hash()
        else:
            # Manual stringification
            from pg.math import MathValue

            if hasattr(result, "student_answer"):
                if isinstance(result.student_answer, MathValue):
                    result.student_answer = result.student_answer.to_string()

            if hasattr(result, "correct_answer"):
                if isinstance(result.correct_answer, MathValue):
                    result.correct_answer = result.correct_answer.to_string()

            if hasattr(result, "preview_latex"):
                if isinstance(result.preview_latex, MathValue):
                    result.preview_latex = result.preview_latex.to_tex()


# Grader registry
_GRADER_REGISTRY: dict[str, ProblemGrader] = {
    "std": std_problem_grader,
    "standard": std_problem_grader,
    "avg": avg_problem_grader,
    "average": avg_problem_grader,
}


def register_grader(name: str, grader: ProblemGrader) -> None:
    """
    Register a custom grader.

    Args:
        name: Grader name
        grader: Grader function
    """
    _GRADER_REGISTRY[name] = grader


def get_grader(name: str) -> ProblemGrader:
    """
    Get grader by name.

    Args:
        name: Grader name

    Returns:
        Grader function

    Raises:
        KeyError: If grader not found
    """
    return _GRADER_REGISTRY[name]
