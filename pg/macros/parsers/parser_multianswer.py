"""
MultiAnswer Parser - Multi-part answer with custom checker.

Allows multiple answer blanks to be checked together with a custom
validation function.

Reference: macros/parsers/parserMultiAnswer.pl
"""

from __future__ import annotations

from typing import Any, Callable


class MultiAnswer:
    """
    Multi-part answer object with custom checker.

    Allows multiple related answers to be checked together,
    enabling validation across multiple input fields.

    Reference: parserMultiAnswer.pl::MultiAnswer
    """

    def __init__(self, *correct_answers):
        """
        Create a multi-part answer.

        Args:
            *correct_answers: Correct values for each part

        Usage:
            multians = MultiAnswer(3, 4, 5)
        """
        self.correct_answers = list(correct_answers)
        self.num_answers = len(correct_answers)

        # Options
        self.checker = None
        self.single_result = False
        self.named_rules = False
        self.checkTypes = True
        self.allowBlankAnswers = False
        self.separator = ", "
        self.tex_separator = ", "

        # Answer blank counter
        self._blank_count = 0

        # Custom messages for each answer blank (set by checker)
        self.messages = {}

    def with_params(self, **options) -> MultiAnswer:
        """
        Set options for the multi-answer.

        Args:
            **options: Options to set
                - checker: Custom checker function
                - singleResult: Show single result for all parts
                - checkTypes: Check answer types
                - allowBlankAnswers: Allow blank answers
                - separator: Separator for display
                - tex_separator: Separator for TeX

        Returns:
            Self for chaining

        Usage:
            multians = MultiAnswer(a, b).with_(
                checker=lambda correct, student, self: ...,
                singleResult=True
            )

        Reference: parserMultiAnswer.pl::with
        """
        if 'checker' in options:
            self.checker = options['checker']
        if 'singleResult' in options:
            self.single_result = options['singleResult']
        if 'checkTypes' in options:
            self.checkTypes = options['checkTypes']
        if 'allowBlankAnswers' in options:
            self.allowBlankAnswers = options['allowBlankAnswers']
        if 'separator' in options:
            self.separator = options['separator']
        if 'tex_separator' in options:
            self.tex_separator = options['tex_separator']

        return self

    def setMessage(self, blank_index: int, message: str) -> None:
        """
        Set a custom message for a specific answer blank.

        Args:
            blank_index: Index of the answer blank (1-indexed)
            message: Custom message to display

        Usage:
            ma.setMessage(1, "Check your arithmetic")

        Reference: parserMultiAnswer.pl::setMessage
        """
        self.messages[blank_index] = message

    def ans_rule(self, width: int = 20) -> str:
        """
        Generate an answer blank for this multi-answer.

        Args:
            width: Width of the input field

        Returns:
            HTML input element

        Reference: parserMultiAnswer.pl::ans_rule
        """
        # Generate unique answer name
        self._blank_count += 1
        name = f"AnSwEr{self._blank_count:04d}"

        # Generate HTML input
        html = f'<input type="text" name="{name}" id="{name}" '
        html += f'size="{width}" style="width: {width * 10}px;" '
        html += f'aria-label="Answer {self._blank_count}">'

        return html

    def cmp(self) -> MultiAnswerEvaluator:
        """
        Return answer evaluator for this multi-answer.

        Returns:
            MultiAnswerEvaluator object

        Reference: parserMultiAnswer.pl::cmp
        """
        return MultiAnswerEvaluator(self)

    def __str__(self):
        return f"MultiAnswer({', '.join(str(x) for x in self.correct_answers)})"


class MultiAnswerEvaluator:
    """
    Evaluator for multi-part answers.

    Collects student answers and validates them using the checker function.
    """

    def __init__(self, multianswer: MultiAnswer):
        """
        Create evaluator for a MultiAnswer.

        Args:
            multianswer: The MultiAnswer object to evaluate
        """
        self.multianswer = multianswer

    def evaluate(self, *student_answers) -> dict:
        """
        Evaluate student answers.

        Args:
            *student_answers: Student's answers for each part

        Returns:
            Dictionary with evaluation results
        """
        ma = self.multianswer

        # Default: all parts correct if they match
        if ma.checker is None:
            # Simple equality check
            if len(student_answers) != len(ma.correct_answers):
                return {
                    'correct': False,
                    'score': 0.0,
                    'message': 'Wrong number of answers'
                }

            # Check each part
            all_correct = True
            for student, correct in zip(student_answers, ma.correct_answers):
                if str(student) != str(correct):
                    all_correct = False
                    break

            return {
                'correct': all_correct,
                'score': 1.0 if all_correct else 0.0,
                'message': ''
            }

        # Use custom checker (only reached if ma.checker is not None)
        try:
            # Parse student answers into MathObjects before passing to checker
            # In Perl, student answers are already parsed by Parser::Formula
            # We need to parse them here using Compute() or the correct answer's context
            parsed_student_answers = []
            for i, student_str in enumerate(student_answers):
                if i < len(ma.correct_answers):
                    correct_ans = ma.correct_answers[i]
                    # Get context from correct answer if available
                    context = None
                    if hasattr(correct_ans, 'context'):
                        context = correct_ans.context
                    elif hasattr(correct_ans, 'getContext'):
                        context = correct_ans.getContext()
                    else:
                        from pg.math.context import get_current_context
                        context = get_current_context()
                    
                    # Parse student answer using Compute (like Parser::Formula in Perl)
                    from pg.math.compute import Compute
                    try:
                        parsed = Compute(str(student_str), context)
                        parsed_student_answers.append(parsed)
                    except Exception:
                        # If parsing fails, use string as-is (checker can handle it)
                        parsed_student_answers.append(str(student_str))
                else:
                    parsed_student_answers.append(str(student_str))
            
            # Call checker with (correct, student, self)
            # In Perl: checker->($correct, $student, $self)
            # In Python: checker(correct, student, self)
            # Note: checker should return a list of scores [score1, score2, ...]
            # At this point, ma.checker is guaranteed to be not None (checked at line 174)
            result = ma.checker(ma.correct_answers, parsed_student_answers, ma)
            
            # Debug: if result is None, the checker might have failed silently
            if result is None:
                import warnings
                warnings.warn(f"MultiAnswer checker returned None - checker may have failed silently. Correct: {ma.correct_answers}, Student: {parsed_student_answers}")

            # Handle None result (broken checker from Perl translation)
            if result is None:
                # Fall back to individual answer checking
                return self._check_individual_answers(ma, student_answers)

            # Result can be boolean or dict
            if isinstance(result, bool):
                return {
                    'correct': result,
                    'score': 1.0 if result else 0.0,
                    'message': ''
                }
            elif isinstance(result, (int, float)):
                correct = result == 1 or result == 1.0
                return {
                    'correct': correct,
                    'score': float(result),
                    'message': ''
                }
            elif isinstance(result, (list, tuple)):
                # List of scores for each answer
                results = [float(r) if isinstance(r, (int, float)) else (1.0 if r else 0.0) for r in result]
                all_correct = all(r >= 1.0 for r in results) if results else False
                return {
                    'correct': all_correct,
                    'score': 1.0 if all_correct else 0.0,
                    'message': '',
                    'results': results
                }
            else:
                # Assume dict-like
                return result

        except Exception as e:
            # On error, fall back to individual checking
            return self._check_individual_answers(ma, student_answers)
    
    def _check_individual_answers(self, ma: MultiAnswer, student_answers: tuple) -> dict:
        """
        Check each answer individually using its own checker.
        
        This is used as a fallback when the custom checker is None or broken.
        """
        if len(student_answers) != len(ma.correct_answers):
            return {
                'correct': False,
                'score': 0.0,
                'message': f'Wrong number of answers: expected {len(ma.correct_answers)}, got {len(student_answers)}',
                'results': [0.0] * len(ma.correct_answers)
            }
        
        # Parse student answers if they're strings
        parsed_student_answers = []
        for i, student in enumerate(student_answers):
            if isinstance(student, str) and i < len(ma.correct_answers):
                correct_ans = ma.correct_answers[i]
                # Get context from correct answer if available
                context = None
                if hasattr(correct_ans, 'context'):
                    context = correct_ans.context
                elif hasattr(correct_ans, 'getContext'):
                    context = correct_ans.getContext()
                else:
                    from pg.math.context import get_current_context
                    context = get_current_context()
                
                # Parse student answer using Compute
                from pg.math.compute import Compute
                try:
                    parsed = Compute(str(student), context)
                    parsed_student_answers.append(parsed)
                except Exception:
                    parsed_student_answers.append(student)
            else:
                parsed_student_answers.append(student)
        
        results = []
        for correct, student in zip(ma.correct_answers, parsed_student_answers):
            score = 0.0
            # Try to use the answer's own checker
            if hasattr(correct, 'cmp'):
                try:
                    checker = correct.cmp()
                    if hasattr(checker, 'check'):
                        # Checker expects string input - convert Formula to string
                        if hasattr(student, 'to_string'):
                            student_str = student.to_string()
                        elif hasattr(student, '__str__'):
                            student_str = str(student)
                        else:
                            student_str = str(student) if not isinstance(student, str) else student
                        check_result = checker.check(student_str)
                        if isinstance(check_result, dict):
                            score = check_result.get('score', 0.0)
                        elif isinstance(check_result, (int, float)):
                            score = float(check_result)
                        else:
                            # If checker doesn't return expected format, fall through to direct comparison
                            raise ValueError("Checker returned unexpected format")
                    elif callable(checker):
                        # Callable checker (like PopUp)
                        check_result = checker(student)
                        if isinstance(check_result, dict):
                            score = check_result.get('score', 0.0)
                        else:
                            raise ValueError("Checker returned unexpected format")
                    else:
                        # No valid checker, fall through to direct comparison
                        raise ValueError("No valid checker method")
                except Exception:
                    # If checker fails, try direct comparison using compare() method
                    if hasattr(correct, 'compare') and callable(correct.compare):
                        try:
                            score = 1.0 if correct.compare(student) else 0.0
                        except Exception:
                            score = 0.0
                    else:
                        # Fallback to == operator
                        score = 1.0 if correct == student else 0.0
            elif hasattr(correct, 'compare') and callable(correct.compare):
                # MathObject with compare() method - use it!
                try:
                    # Both should be MathObjects at this point
                    if hasattr(student, 'compare') or hasattr(student, '__eq__'):
                        # Use compare() method for proper MathObject comparison
                        score = 1.0 if correct.compare(student) else 0.0
                    else:
                        # Student is not a MathObject, try == operator
                        score = 1.0 if correct == student else 0.0
                except Exception as e:
                    # If comparison fails, answer is wrong
                    score = 0.0
            else:
                # Simple comparison using == operator
                try:
                    score = 1.0 if correct == student else 0.0
                except Exception:
                    score = 0.0
            
            results.append(score)
        
        all_correct = all(r >= 1.0 for r in results)
        return {
            'correct': all_correct,
            'score': 1.0 if all_correct else 0.0,
            'message': '',
            'results': results
        }

    def __str__(self):
        return f"MultiAnswerEvaluator({self.multianswer})"
