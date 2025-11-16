"""
MultiAnswer - Multiple Related Answer Checking

Provides a class for checking multiple related answer blanks together,
allowing for custom cross-answer validation and scoring logic.

Based on WeBWorK's PGcore.pl MultiAnswer macro.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple


class MultiAnswer:
    """
    Stub for MultiAnswer - used for checking multiple related answer blanks together.
    
    Allows custom checkers to validate relationships between multiple answers
    or apply complex scoring logic across multiple answer blanks.
    
    Example:
        ma = MultiAnswer(ans1, ans2, ans3)
        ma.with_params(checker=lambda correct, student, self: check_logic(...))
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Initialize MultiAnswer with correct answers.
        
        Args:
            *args: Correct answer values/evaluators
            **kwargs: Options including 'checker' function
        """
        self.answers = args
        self.options = kwargs

    def with_params(self, **kwargs: Any) -> 'MultiAnswer':
        """
        Set parameters/options (works around 'with' keyword).
        
        Args:
            **kwargs: Options to set (e.g., checker function)
            
        Returns:
            Self for method chaining
        """
        self.options.update(kwargs)
        return self

    def cmp(self) -> 'MultiAnswer':
        """
        Return a checker that can check multiple answers together.
        
        Returns:
            Self (acts as the checker object)
        """
        # For now, return self so it can be used as a checker
        return self

    def check(self, *student_answers: Any) -> Dict[str, Any]:
        """
        Check multiple student answers against the correct answers.
        
        Args:
            *student_answers: Student's answers to check
            
        Returns:
            Dictionary with 'correct', 'score', 'message', and 'results' keys
        """
        # Extract the custom checker if provided
        checker_func = self.options.get('checker')

        if checker_func and callable(checker_func):
            # Call custom checker with (correct, student, self) tuple
            try:
                results = checker_func(self.answers, student_answers, self)
                # results should be a list of [score1, score2, ...]
                # Convert to dict format for each answer
                if isinstance(results, list):
                    # Return results for all answers
                    # For now, return a dict with aggregate result
                    all_correct = all(r >= 1.0 for r in results) if results else False
                    return {
                        'correct': all_correct,
                        'score': 1.0 if all_correct else 0.0,
                        'message': 'Checked with custom MultiAnswer checker',
                        'results': results,  # Individual results for each blank
                    }
            except Exception as e:
                return {
                    'correct': False,
                    'score': 0.0,
                    'message': f'Error in custom checker: {str(e)}',
                }

        # Default: check each answer individually
        if len(student_answers) != len(self.answers):
            return {
                'correct': False,
                'score': 0.0,
                'message': f'Expected {len(self.answers)} answers, got {len(student_answers)}',
            }

        results = []
        for correct, student in zip(self.answers, student_answers):
            if hasattr(correct, 'cmp'):
                checker = correct.cmp()
                if hasattr(checker, 'check'):
                    result = checker.check(student)
                    results.append(result.get('score', 0.0))
                else:
                    results.append(0.0)
            else:
                # Simple comparison
                results.append(1.0 if str(correct) == str(student) else 0.0)

        all_correct = all(r >= 1.0 for r in results)
        return {
            'correct': all_correct,
            'score': 1.0 if all_correct else 0.0,
            'message': '',
            'results': results,
        }


# Add .with() method using setattr to work around Python keyword
setattr(MultiAnswer, 'with', MultiAnswer.with_params)


__all__ = ['MultiAnswer']

