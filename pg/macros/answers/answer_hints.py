r"""
Answer Hints for WeBWorK.

This module provides custom answer hints and feedback for common student errors.
Hints can be triggered by:
1. Specific wrong answer values
2. Arrays of values that all trigger the same message
3. Callable functions that determine whether to show a hint

Reference: macros/answers/answerHints.pl

Example:
    >>> ANS(Vector(1,2,3)->cmp()->withPostFilter(AnswerHints(
    ...     Vector(0,0,0) => "The zero vector is not valid",
    ...     ["<1,1,1>","<2,2,2>"] => "Don't just guess!",
    ...     sub { correct, student, ans -> ... } => "Your answer is close",
    ... )))
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def AnswerHints(*args: Any) -> Callable:
    r"""
    Creates a post-filter for answer hints that provides custom feedback.

    Hint triggers can be:
    1. A specific value: Vector(0,0,0) => "message"
    2. Array of values: [value1, value2] => "message"
    3. Callable function: (lambda correct, student, ans: ...) => "message"

    Message can be:
    - A string: "error message"
    - Array with options: ["message", option_key => option_value, ...]

    Options:
    - checkCorrect (bool): Check hints even if answer is correct (default: False)
    - replaceMessage (bool): OK to replace existing message (default: False)
    - checkTypes (bool): Only check if student answer is same type (default: True)
    - processPreview (bool): Process during answer preview (default: False)
    - score (float): Score to assign if hint triggered (default: keep original)
    - cmp_options (list): Options for compare function (default: [])

    Args:
        *args: Variable length of (trigger, message) pairs

    Returns:
        A post-filter function for answer checkers

    Example:
        >>> AnswerHints(
        ...     Vector(0,0,0) => "The zero vector is not valid",
        ...     ["<1,1,1>","<2,2,2>"] => "Don't just guess!",
        ...     lambda c, s, a: abs(c - s) < 0.1 => ["Close!", score => 0.25],
        ... )

    Reference: macros/answers/answerHints.pl
    """
    # Parse the hint arguments as (trigger, message) pairs
    hints_list: List[Tuple[Any, str, Dict[str, Any]]] = []

    i = 0
    while i < len(args):
        if i + 1 >= len(args):
            break

        trigger = args[i]
        message_spec = args[i + 1]

        # Parse message and options
        message = message_spec
        options = {}

        if isinstance(message_spec, (list, tuple)):
            # Message with options: ["message", key => value, ...]
            message = message_spec[0]
            # Parse the remaining elements as key-value pairs
            for j in range(1, len(message_spec), 2):
                if j + 1 < len(message_spec):
                    options[message_spec[j]] = message_spec[j + 1]

        # Set defaults for options
        opts = {
            'checkCorrect': options.get('checkCorrect', False),
            'replaceMessage': options.get('replaceMessage', False),
            'checkTypes': options.get('checkTypes', True),
            'processPreview': options.get('processPreview', False),
            'score': options.get('score', None),
            'cmp_options': options.get('cmp_options', []),
        }

        hints_list.append((trigger, message, opts))
        i += 2

    def post_filter(ans: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-filter function that checks hints and adds messages to answers.

        Args:
            ans: Answer hash with keys: correct_value, student_value, score,
                 ans_message, error_message, isPreview, etc.

        Returns:
            Modified answer hash with hint messages added if applicable
        """
        ans['_filter_name'] = 'Answer Hints Post Filter'

        # Get correct and student values
        correct = ans.get('correct_value')
        student = ans.get('student_value')

        # Both must be MathObjects (have ref)
        if not (hasattr(correct, '__class__') and hasattr(student, '__class__')):
            return ans

        # Student answer must exist and be a value
        if student is None:
            return ans

        # Get preview flag
        is_preview = ans.get('isPreview', False)

        # Check each hint in order
        for trigger, message, opts in hints_list:
            # Skip if processing preview but we shouldn't
            if is_preview and not opts['processPreview']:
                continue

            # Skip if answer is correct and we shouldn't check correct answers
            score = ans.get('score', 0)
            if score >= 1 and not opts['checkCorrect']:
                continue

            # Skip if there's already a message and we can't replace it
            ans_message = ans.get('ans_message', '')
            if ans_message and not opts['replaceMessage']:
                continue

            # Check the trigger
            hint_matched = False

            if callable(trigger):
                # Callable trigger: run the function
                # Check type compatibility if required
                if opts['checkTypes']:
                    try:
                        correct_type = getattr(correct, 'type', None)
                        student_type = getattr(student, 'type', None)
                        if correct_type != student_type:
                            continue
                    except Exception:
                        pass

                # Call the trigger function with (correct, student, ans)
                try:
                    hint_matched = trigger(correct, student, ans)
                except Exception:
                    # Silently fail on errors in trigger function
                    pass
            else:
                # Value or array trigger: compare with student answer
                # Normalize trigger to list
                trigger_list = trigger if isinstance(trigger, (list, tuple)) else [trigger]

                # Check each value in the trigger list
                for wrong_value in trigger_list:
                    # Convert string answers to Formula if needed
                    if isinstance(wrong_value, str):
                        try:
                            # Try to parse as Formula
                            wrong_value = _parse_formula(wrong_value)
                        except Exception:
                            pass

                    # Compare wrong_value with student answer
                    if _compare_answers(wrong_value, student, ans, opts['cmp_options']):
                        hint_matched = True
                        break

            # If hint was matched, add the message and potentially override score
            if hint_matched:
                ans['ans_message'] = message
                ans['error_message'] = message
                if opts['score'] is not None:
                    ans['score'] = opts['score']
                break  # Only use first matching hint

        return ans

    return post_filter


def _parse_formula(formula_str: str) -> Any:
    """
    Parse a string formula into a Formula object.

    Args:
        formula_str: String representation of formula

    Returns:
        Parsed formula object
    """
    # This is a placeholder - in actual WeBWorK environment,
    # this would use the Formula class from MathObjects
    # For now, return a simple object that can be compared
    return {'_formula_str': formula_str, '_is_formula': True}


def _compare_answers(
    expected: Any,
    student: Any,
    ans: Dict[str, Any],
    cmp_options: List[Any]
) -> bool:
    """
    Compare two answers using comparison logic.

    Uses the MathObject comparison system similar to cmp() in Perl.

    Args:
        expected: The expected answer value
        student: The student's answer value
        ans: The answer hash (for context)
        cmp_options: Options for comparison (passed to cmp)

    Returns:
        True if answers match, False otherwise
    """
    # Check if both are the same object
    if expected is student:
        return True

    # Check if they have the same string representation
    expected_str = getattr(expected, 'string', str(expected))
    student_str = getattr(student, 'string', str(student))

    if expected_str == student_str:
        return True

    # Try to use cmp() method if available
    try:
        if hasattr(expected, 'cmp'):
            # Create a copy of ans for comparison
            test_ans = {
                'correct_value': expected,
                'student_value': student,
                'score': 0,
                'ans_message': '',
                'error_message': '',
            }

            # Run comparison preprocessor
            if hasattr(expected, 'cmp_preprocess'):
                expected.cmp_preprocess(test_ans)

            # Run comparison checker
            if hasattr(expected, 'cmp_equal'):
                expected.cmp_equal(test_ans)

            # Run postprocessor if no errors
            if not test_ans.get('error_message') and hasattr(expected, 'cmp_postprocess'):
                expected.cmp_postprocess(test_ans)

            return test_ans.get('score', 0) >= 1
    except Exception:
        pass

    # Default: no match
    return False


__all__ = [
    'AnswerHints',
]

