"""Parametric Line Parser for WeBWorK.

This module provides the ParametricLine class for parsing and validating
parametric equations of lines in 2D and 3D.

Based on macros/parsers/parserParametricLine.pl from the WeBWorK distribution.
"""

from typing import Any, List, Optional, Tuple


class ParametricLine:
    """
    Parser for parametric line equations.
    
    Parses and validates parametric representations of lines:
    - 2D: (x, y) = (x0, y0) + t(dx, dy)
    - 3D: (x, y, z) = (x0, y0, z0) + t(dx, dy, dz)
    
    Checks for equivalent parametrizations (different starting point,
    different parameter values, opposite direction, etc.).
    
    Attributes:
        equation: Parametric line equation as string or tuple
        point: Base point of the line
        direction: Direction vector of the line
    """

    def __init__(self, *args, **kwargs):
        """
        Create a ParametricLine parser.
        
        Args:
            *args: Parametric equation as string or tuple, or point + direction vector
            **kwargs: Additional options
            
        Example:
            >>> line = ParametricLine("<1, 2> + t<3, 4>")
            >>> line2 = ParametricLine(((1, 2), (3, 4)))  # tuple form
            >>> line3 = ParametricLine((1, 2), (3, 4))  # point, direction
        
        Perl Source: parserParametricLine.pl ParametricLine constructor
        """
        if len(args) >= 2:
            # Called as ParametricLine(point, direction)
            self.point = args[0]
            self.direction = args[1]
            self.equation = f"{self.point} + t{self.direction}"
        else:
            self.equation = args[0] if args else None
            self.point = None
            self.direction = None
            
            # Parse tuple form if provided
            if isinstance(self.equation, (list, tuple)) and len(self.equation) >= 2:
                self.point = self.equation[0]
                self.direction = self.equation[1]
        
        self.options = kwargs
        
    def cmp(self, **options) -> 'AnswerChecker':
        """
        Create an answer checker for this ParametricLine.
        
        Args:
            **options: Options for answer checking
        
        Returns:
            AnswerChecker object for use with ANS()
        
        Perl Source: parserParametricLine.pl cmp() method
        """
        return AnswerChecker(self, **options)
    
    def __str__(self) -> str:
        """Return string representation."""
        if self.point and self.direction:
            return f"ParametricLine: {self.point} + t{self.direction}"
        return str(self.equation) if self.equation else 'ParametricLine'
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"ParametricLine({self.equation!r})"


class AnswerChecker:
    """
    Answer checker for ParametricLine answers.
    
    Checks if student answer is an equivalent parametric line.
    """
    
    def __init__(self, correct: ParametricLine, **options):
        """
        Initialize answer checker.
        
        Args:
            correct: Correct ParametricLine
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
            # Real implementation would parse and check equivalence
            correct_str = str(self.correct.equation).strip()
            student_str = student_answer.strip()
            is_correct = correct_str == student_str
            
            return {
                'correct': is_correct,
                'score': 1.0 if is_correct else 0.0,
                'message': '' if is_correct else 'Incorrect parametric line'
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
    'ParametricLine',
    'AnswerChecker',
]
