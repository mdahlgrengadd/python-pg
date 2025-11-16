"""
PopUp and Dropdown Answer Parsers

Provides PopUp, DropDown, DropDownTF, and RadioButtons classes for multiple-choice
style answer checking.

Based on WeBWorK's PG macro libraries (parserPopUp.pl).
"""

from typing import Any, Callable, Dict, List, Optional


class PopUp:
    """PopUp/DropDown menu for multiple choice questions."""

    def __init__(self, choices: List[str], correct: Any, **options: Any):
        """
        Initialize PopUp with choices and correct answer.
        
        Args:
            choices: List of choices to display
            correct: Correct choice value
            **options: Additional options
        """
        self.choices = choices
        self.correct = correct
        self.options = options

    def menu(self) -> str:
        """
        Generate HTML for the dropdown menu.
        
        Returns:
            HTML string for select element
        """
        options = []
        for choice in self.choices:
            # Escape HTML in choices
            choice_str = str(choice).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
            options.append(f'<option value="{choice_str}">{choice_str}</option>')
        
        return f'<select class="pg-popup-menu">{"".join(options)}</select>'
    
    def cmp(self) -> Callable:
        """
        Return a checker function for this PopUp.
        
        Returns:
            Function that checks student answer
        """
        correct = self.correct
        return lambda x: {
            'correct': str(x) == str(correct),
            'score': 1.0 if str(x) == str(correct) else 0.0
        }


class DropDown(PopUp):
    """Alias for PopUp - DropDown menu."""
    pass


class DropDownTF:
    """DropDown for True/False questions."""

    def __init__(self, correct: bool | str, **options: Any):
        """
        Initialize DropDownTF with correct answer.
        
        Args:
            correct: True/False or 'T'/'F' correct answer
            **options: Additional options
        """
        # Normalize: accept 'T'/'F' or True/False
        if isinstance(correct, str):
            self.correct = correct.upper()  # 'T' or 'F'
        else:
            self.correct = 'T' if correct else 'F'
        self.choices = ['True', 'False']
        self.options = options

    def cmp(self) -> Callable:
        """
        Return a checker function for this DropDownTF.
        
        Returns:
            Function that checks student answer
        """
        correct = self.correct
        # Handle both 'T'/'F' format and 'True'/'False' format
        # Map student input to canonical form
        def normalize_answer(ans):
            ans_str = str(ans).strip()
            if ans_str.upper() in ('T', 'TRUE'):
                return 'True'
            elif ans_str.upper() in ('F', 'FALSE'):
                return 'False'
            return ans_str
        
        # Map correct answer to 'True' or 'False'
        if correct.upper() in ('T', 'TRUE'):
            correct_normalized = 'True'
        else:
            correct_normalized = 'False'
        
        return lambda x: {
            'correct': normalize_answer(x) == correct_normalized,
            'score': 1.0 if normalize_answer(x) == correct_normalized else 0.0
        }


class RadioButtons:
    """Radio buttons for multiple choice questions."""

    def __init__(self, choices: List[str], correct: Any, **options: Any):
        """
        Initialize RadioButtons with choices and correct answer.
        
        Args:
            choices: List of choices to display
            correct: Correct choice value
            **options: Additional options
        """
        self.choices = choices
        self.correct = correct
        self.options = options

    def cmp(self) -> Callable:
        """
        Return a checker function for these RadioButtons.
        
        Returns:
            Function that checks student answer
        """
        correct = self.correct
        return lambda x: {
            'correct': str(x) == str(correct),
            'score': 1.0 if str(x) == str(correct) else 0.0
        }


__all__ = ['PopUp', 'DropDown', 'DropDownTF', 'RadioButtons']
