"""
PGchoicemacros.pl - Multiple choice, true/false, matching questions

DEPRECATED: This module is deprecated. Use pg.macros.choice instead.

The correct implementations are in pg/macros/choice/pg_choice_macros.py

Reference: macros/ui/PGchoicemacros.pl (1,089 lines)
"""

import warnings
import random
from typing import Any, List, Optional

warnings.warn(
    "pg.macros.ui.choice_macros is deprecated. Use pg.macros.choice instead.",
    DeprecationWarning,
    stacklevel=2
)


class MultipleChoice:
    """
    Multiple choice question (radio buttons).
    
    Reference: PGchoicemacros.pl::new_multiple_choice
    """
    
    def __init__(self):
        self.question = ""
        self.choices = []
        self.correct_answer = None
        self.extra_answers = []
    
    def qa(self, question: str, correct_answer: str) -> "MultipleChoice":
        """Set question and correct answer."""
        self.question = question
        self.correct_answer = correct_answer
        return self
    
    def extra(self, *incorrect_answers: str) -> "MultipleChoice":
        """Add incorrect answer choices."""
        self.extra_answers.extend(incorrect_answers)
        return self
    
    def makeLast(self, *choices: str) -> "MultipleChoice":
        """Make specific choices appear last."""
        # Remove from extras if present
        for choice in choices:
            if choice in self.extra_answers:
                self.extra_answers.remove(choice)
        # Will be added at end
        self.extra_answers.extend(choices)
        return self
    
    def print_q(self) -> str:
        """Print the question."""
        return self.question
    
    def print_a(self) -> str:
        """Print answer choices as HTML radio buttons."""
        # Combine correct and incorrect answers
        all_choices = [self.correct_answer] + self.extra_answers
        # Shuffle (keeping last items if makeLast was used)
        random.shuffle(all_choices[:-len(self.extra_answers) if self.extra_answers else len(all_choices)])
        
        html = '<div class="multiple-choice">\n'
        for i, choice in enumerate(all_choices):
            html += f'  <label><input type="radio" name="choice" value="{i}"> {choice}</label><br>\n'
        html += '</div>'
        return html
    
    def correct_ans(self) -> str:
        """Return the correct answer text."""
        return self.correct_answer


class CheckboxMultipleChoice:
    """
    Multiple choice with checkboxes (multiple answers).
    
    Reference: PGchoicemacros.pl::new_checkbox_multiple_choice
    """
    
    def __init__(self):
        self.question = ""
        self.correct_answers = []
        self.incorrect_answers = []
    
    def qa(self, question: str, *correct_answers: str) -> "CheckboxMultipleChoice":
        """Set question and correct answers."""
        self.question = question
        self.correct_answers = list(correct_answers)
        return self
    
    def extra(self, *incorrect_answers: str) -> "CheckboxMultipleChoice":
        """Add incorrect answer choices."""
        self.incorrect_answers.extend(incorrect_answers)
        return self
    
    def print_q(self) -> str:
        """Print the question."""
        return self.question
    
    def print_a(self) -> str:
        """Print answer choices as HTML checkboxes."""
        # Combine and shuffle
        all_choices = self.correct_answers + self.incorrect_answers
        random.shuffle(all_choices)
        
        html = '<div class="checkbox-multiple-choice">\n'
        for i, choice in enumerate(all_choices):
            html += f'  <label><input type="checkbox" name="choice" value="{i}"> {choice}</label><br>\n'
        html += '</div>'
        return html
    
    def correct_ans(self) -> List[str]:
        """Return list of correct answer texts."""
        return self.correct_answers


class TrueFalse:
    """
    True/False question.
    
    Reference: PGchoicemacros.pl::new_true_false
    """
    
    def __init__(self, question: str, answer: bool):
        self.question = question
        self.answer = answer
    
    def print_q(self) -> str:
        """Print the question."""
        return self.question
    
    def print_a(self) -> str:
        """Print True/False radio buttons."""
        html = '<div class="true-false">\n'
        html += '  <label><input type="radio" name="tf" value="T"> True</label><br>\n'
        html += '  <label><input type="radio" name="tf" value="F"> False</label><br>\n'
        html += '</div>'
        return html
    
    def correct_ans(self) -> str:
        """Return 'T' or 'F'."""
        return 'T' if self.answer else 'F'


class PopUp:
    """
    Dropdown/popup menu.
    
    Reference: PGchoicemacros.pl::new_pop_up_select_list
    """
    
    def __init__(self, choices: List[str]):
        self.choices = choices
        self.correct_answer = None
    
    def qa(self, *pairs) -> "PopUp":
        """Set question-answer pairs."""
        # First item is correct
        if pairs:
            self.correct_answer = pairs[0] if isinstance(pairs[0], str) else pairs[0][1]
        return self
    
    def print_q(self) -> str:
        """Print as HTML select."""
        html = '<select name="popup">\n'
        html += '  <option value="">?</option>\n'
        for choice in self.choices:
            html += f'  <option value="{choice}">{choice}</option>\n'
        html += '</select>'
        return html
    
    def correct_ans(self) -> str:
        """Return correct answer."""
        return self.correct_answer or ""


class MatchList:
    """
    Matching question.
    
    Reference: PGchoicemacros.pl::new_match_list
    """
    
    def __init__(self):
        self.questions = []
        self.answers = []
        self.pairs = []
    
    def qa(self, *pairs) -> "MatchList":
        """Set question-answer pairs."""
        for i in range(0, len(pairs), 2):
            if i + 1 < len(pairs):
                self.pairs.append((pairs[i], pairs[i + 1]))
                self.questions.append(pairs[i])
                if pairs[i + 1] not in self.answers:
                    self.answers.append(pairs[i + 1])
        return self
    
    def extra(self, *extra_answers: str) -> "MatchList":
        """Add extra answer choices."""
        for ans in extra_answers:
            if ans not in self.answers:
                self.answers.append(ans)
        return self
    
    def print_q(self) -> str:
        """Print matching questions."""
        html = '<div class="match-list">\n'
        random.shuffle(self.answers)
        
        for i, question in enumerate(self.questions):
            html += f'<div class="match-item">\n'
            html += f'  <span class="question">{i + 1}. {question}</span>\n'
            html += f'  <select name="match_{i}">\n'
            html += '    <option value="">?</option>\n'
            for j, answer in enumerate(self.answers):
                html += f'    <option value="{j}">{chr(65 + j)}. {answer}</option>\n'
            html += '  </select>\n'
            html += '</div>\n'
        
        html += '</div>'
        return html
    
    def print_a(self) -> str:
        """Print answer list."""
        html = '<ol type="A">\n'
        for answer in self.answers:
            html += f'  <li>{answer}</li>\n'
        html += '</ol>'
        return html
    
    def correct_ans(self) -> List[str]:
        """Return list of correct answers."""
        correct = []
        for question, answer in self.pairs:
            try:
                idx = self.answers.index(answer)
                correct.append(chr(65 + idx))
            except ValueError:
                correct.append("?")
        return correct


# Factory functions (Perl-compatible API)

def new_multiple_choice() -> MultipleChoice:
    """Create new multiple choice question."""
    return MultipleChoice()


def new_checkbox_multiple_choice() -> CheckboxMultipleChoice:
    """Create new checkbox multiple choice question."""
    return CheckboxMultipleChoice()


def new_true_false(question: str, answer: bool) -> TrueFalse:
    """Create new true/false question."""
    return TrueFalse(question, answer)


def new_pop_up_select_list(choices: List[str]) -> PopUp:
    """Create new popup select list."""
    return PopUp(choices)


def new_match_list() -> MatchList:
    """Create new matching question."""
    return MatchList()


