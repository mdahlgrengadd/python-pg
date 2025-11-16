"""Solution For Parser for WeBWorK.

This module provides the SolutionFor class for checking if an expression
is a solution to a differential equation or other equation.

Based on macros/parsers/parserSolutionFor.pl from the WeBWorK distribution.
"""

from typing import Any, Dict, Optional


class SolutionFor:
    """
    Parser for checking solutions to equations.
    
    Checks if an expression is a valid solution to a given equation,
    particularly useful for differential equation problems.
    
    Attributes:
        formula: The solution formula as a Formula object or string
        equation: The equation being solved (optional, for reference)
    """

    def __init__(self, formula: Any = None, equation: Any = None, **kwargs):
        """
        Create a SolutionFor parser.
        
        Args:
            formula: The formula that should be a solution
            equation: The equation being solved (optional)
            **kwargs: Additional options
            
        Example:
            >>> sol = SolutionFor("e^x", "dy/dx = y")
        
        Perl Source: parserSolutionFor.pl SolutionFor constructor
        """
        self.formula = formula
        self.equation = equation
        self.options = kwargs
        
    def cmp(self, **options) -> 'AnswerChecker':
        """
        Create an answer checker for this SolutionFor.
        
        Args:
            **options: Options for answer checking
        
        Returns:
            AnswerChecker object for use with ANS()
        
        Perl Source: parserSolutionFor.pl cmp() method
        """
        return AnswerChecker(self, **options)
    
    def __getitem__(self, key: str) -> Any:
        """Support dict-like access - returns formula for 'f' key."""
        if key == 'f':
            return self.formula
        elif key == 'solution':
            return self.formula
        raise KeyError(key)
    
    def __str__(self) -> str:
        """Return string representation."""
        return str(self.formula) if self.formula else ''
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"SolutionFor({self.formula!r}, {self.equation!r})"


class AnswerChecker:
    """
    Answer checker for SolutionFor answers.
    
    Checks if student answer is a valid solution to the equation.
    """
    
    def __init__(self, correct: SolutionFor, **options):
        """
        Initialize answer checker.
        
        Args:
            correct: Correct SolutionFor
            **options: Checking options
        """
        self.correct = correct
        self.options = options
    
    def check(self, student_answer: str) -> dict:
        """
        Check a student answer.
        
        Args:
            student_answer: Student's answer as string
        
        Returns:
            Dict with keys: correct (bool), score (float), message (str)
        """
        try:
            # Simple string comparison for now
            # Real implementation would parse and verify the solution
            is_correct = str(self.correct.formula).strip() == student_answer.strip()
            
            return {
                'correct': is_correct,
                'score': 1.0 if is_correct else 0.0,
                'message': '' if is_correct else 'Not a valid solution'
            }
        except Exception as e:
            return {
                'correct': False,
                'score': 0.0,
                'message': f'Error checking answer: {str(e)}'
            }
    
    def __call__(self, **kwargs):
        """Make checker callable - returns self for chaining."""
        return self


__all__ = [
    'SolutionFor',
    'AnswerChecker',
]
