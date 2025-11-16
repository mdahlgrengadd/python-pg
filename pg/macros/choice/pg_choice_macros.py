"""
PGchoicemacros.pl - Multiple choice, true/false, matching

Python port of macros/core/PGchoicemacros.pl
Provides multiple choice question utilities.

Reference: PGchoicemacros.pl
"""

from dataclasses import dataclass
from typing import Any

from pg.answer import AnswerEvaluator
from pg.answer.evaluators.string import StringEvaluator

# Export list
__exports__ = [
    "MultipleChoice",
    "CheckboxMultipleChoice",
    "TrueFalse",
    "new_multiple_choice",
    "new_checkbox_multiple_choice",
    "new_true_false",
]


@dataclass
class MultipleChoice:
    """
    Multiple choice question.

    Reference: PGchoicemacros.pl::new_multiple_choice
    """

    question: str = ""
    choices: list[str] = None
    correct_index: int = 0

    def __post_init__(self):
        if self.choices is None:
            self.choices = []

    def qa(self, question: str, correct: str) -> None:
        """
        Set question and correct answer.

        Args:
            question: Question text
            correct: Correct answer
        """
        self.question = question
        self.choices = [correct]
        self.correct_index = 0

    def extra(self, *incorrect: str) -> None:
        """
        Add incorrect answer choices.

        Args:
            *incorrect: Wrong answers to add
        """
        self.choices.extend(incorrect)

    def makeLast(self, *choices: str) -> None:
        """
        Force choices to appear last.

        Args:
            *choices: Choices to put at end
        """
        # Move specified choices to end
        for choice in choices:
            if choice in self.choices:
                self.choices.remove(choice)
                self.choices.append(choice)

    def shuffle(self, seed: int | None = None) -> None:
        """
        Shuffle choices (except those marked with makeLast).

        Args:
            seed: Random seed
        """
        import random

        if seed is not None:
            random.seed(seed)

        # Get correct answer
        correct = self.choices[self.correct_index]

        # Shuffle
        random.shuffle(self.choices)

        # Update correct index
        self.correct_index = self.choices.index(correct)

    def print_q(self) -> str:
        """
        Return question HTML.

        Returns:
            Question HTML
        """
        return f"<p>{self.question}</p>"

    def print_a(self) -> str:
        """
        Return choices HTML (radio buttons).

        Returns:
            Choices HTML
        """
        html = []
        for i, choice in enumerate(self.choices):
            label = chr(65 + i)  # A, B, C, ...
            html.append(
                f'<div class="choice">'
                f'<input type="radio" name="answer" value="{i}" id="choice_{i}">'
                f'<label for="choice_{i}">{label}. {choice}</label>'
                f'</div>'
            )
        return "\n".join(html)

    def correct_ans(self) -> str:
        """
        Return correct answer as choice letter (A, B, C, ...).

        Returns:
            Correct choice letter
        """
        return chr(65 + self.correct_index)  # A=65 in ASCII

    def cmp(self) -> AnswerEvaluator:
        """
        Return answer evaluator.

        Returns:
            Radio button answer evaluator
        """
        from pg.answer.cmp import radio_cmp
        return radio_cmp(self.correct_index)


@dataclass
class TrueFalse:
    """
    True/False question.

    Reference: PGchoicemacros.pl::new_true_false
    """

    question: str = ""
    correct: bool = True

    def qa(self, question: str, correct: bool | str) -> None:
        """
        Set question and answer.

        Args:
            question: Question text
            correct: Correct answer (True/False or "T"/"F")
        """
        self.question = question

        if isinstance(correct, str):
            self.correct = correct.upper() in ("T", "TRUE")
        else:
            self.correct = correct

    def print_q(self) -> str:
        """Return question HTML."""
        return f"<p>{self.question}</p>"

    def print_a(self) -> str:
        """Return choices HTML."""
        return """
<div class="choice">
    <input type="radio" name="answer" value="T" id="choice_true">
    <label for="choice_true">True</label>
</div>
<div class="choice">
    <input type="radio" name="answer" value="F" id="choice_false">
    <label for="choice_false">False</label>
</div>
"""

    def correct_ans(self) -> str:
        """Return correct answer."""
        return "T" if self.correct else "F"

    def cmp(self) -> AnswerEvaluator:
        """Return answer evaluator."""
        return StringEvaluator(
            correct_answer=self.correct_ans(),
            case_sensitive=False,
        )


def new_multiple_choice() -> MultipleChoice:
    """
    Create new multiple choice question.

    Returns:
        MultipleChoice object

    Reference: PGchoicemacros.pl::new_multiple_choice
    """
    return MultipleChoice()


def new_true_false() -> TrueFalse:
    """
    Create new true/false question.

    Returns:
        TrueFalse object

    Reference: PGchoicemacros.pl::new_true_false
    """
    return TrueFalse()


@dataclass
class CheckboxMultipleChoice:
    """
    Checkbox multiple choice (multiple correct answers).

    Reference: PGchoicemacros.pl::new_checkbox_multiple_choice
    """

    question: str = ""
    correct_choices: list[str] = None
    incorrect_choices: list[str] = None

    def __post_init__(self):
        if self.correct_choices is None:
            self.correct_choices = []
        if self.incorrect_choices is None:
            self.incorrect_choices = []

    def qa(self, question: str, *correct: str) -> None:
        """
        Set question and correct answers.

        Args:
            question: Question text
            *correct: Correct answers
        """
        self.question = question
        self.correct_choices = list(correct)

    def extra(self, *incorrect: str) -> None:
        """
        Add incorrect choices.

        Args:
            *incorrect: Incorrect answers
        """
        self.incorrect_choices.extend(incorrect)

    def print_q(self) -> str:
        """
        Return question HTML.

        Returns:
            Question HTML
        """
        return f"<p>{self.question}</p>"

    def print_a(self) -> str:
        """
        Return choices as checkboxes.

        Returns:
            Checkbox HTML
        """
        all_choices = self.correct_choices + self.incorrect_choices
        html = []
        for i, choice in enumerate(all_choices):
            label = chr(65 + i)  # A, B, C, ...
            html.append(
                f'<div class="choice">'
                f'<input type="checkbox" name="answer" value="{i}" id="choice_{i}">'
                f'<label for="choice_{i}">{label}. {choice}</label>'
                f'</div>'
            )
        return "\n".join(html)

    def correct_ans(self) -> list[str]:
        """
        Return correct answer letters.

        Returns:
            List of correct choice letters
        """
        return [chr(65 + i) for i in range(len(self.correct_choices))]

    def cmp(self) -> AnswerEvaluator:
        """
        Return checkbox answer evaluator.

        Returns:
            Checkbox answer evaluator
        """
        from pg.answer.cmp import checkbox_cmp
        indices = list(range(len(self.correct_choices)))
        return checkbox_cmp(indices)


def new_checkbox_multiple_choice() -> CheckboxMultipleChoice:
    """
    Create new checkbox multiple choice question.

    Returns:
        CheckboxMultipleChoice object

    Reference: PGchoicemacros.pl::new_checkbox_multiple_choice
    """
    return CheckboxMultipleChoice()
