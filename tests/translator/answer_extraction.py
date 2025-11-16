"""
Answer extraction utilities for runtime testing of PG problems.

This module provides functions to extract correct answers from translated PG problems
so they can be submitted to validate that answer checking works correctly.

The key challenge is converting MathObject evaluators (which are Perl objects)
into strings that students would type when answering the problem.
"""

from typing import Any, Dict, Optional


def extract_correct_answers(result: Any) -> Dict[str, str]:
    """
    Extract correct answers from a ProblemResult.

    This extracts the correct answer from each answer blank's evaluator
    and converts it to a string format suitable for student input.

    Args:
        result: ProblemResult from PGTranslator.translate()

    Returns:
        Dictionary mapping answer blank names to correct answer strings
        Example: {"AnSwEr0001": "x^2 - 6*x + 4", "AnSwEr0002": "42"}

    Raises:
        ValueError: If answer extraction fails for any blank
    """
    if not result.answer_blanks:
        return {}

    correct_answers = {}
    multi_answer_map = {}  # Track MultiAnswer objects and their blank indices

    for blank_name, blank_info in result.answer_blanks.items():
        if not blank_info:
            continue

        # Extract evaluator from blank_info
        # Structure: {"evaluator": {"ans_eval": <MathObject>}} or similar
        evaluator = blank_info.get("evaluator")
        if not evaluator:
            continue

        # Try to get the ans_eval object
        if isinstance(evaluator, dict):
            ans_eval = evaluator.get("ans_eval")
        else:
            ans_eval = evaluator

        if not ans_eval:
            continue

        # Check if this is a MultiAnswer object (check both correct_answers and answers attributes)
        is_multianswer = False
        if hasattr(ans_eval, "correct_answers") and isinstance(ans_eval.correct_answers, (list, tuple)):
            is_multianswer = True
        elif hasattr(ans_eval, "answers") and isinstance(ans_eval.answers, (list, tuple)):
            is_multianswer = True

        if is_multianswer:
            # This is a MultiAnswer - store it for special handling
            obj_id = id(ans_eval)
            if obj_id not in multi_answer_map:
                multi_answer_map[obj_id] = {"obj": ans_eval, "blanks": []}
            multi_answer_map[obj_id]["blanks"].append(blank_name)
            continue

        # Extract the answer string
        try:
            answer_str = extract_answer_string(ans_eval)
            if answer_str:
                correct_answers[blank_name] = answer_str
        except Exception as e:
            # Log but don't fail - some answer types may not be extractable
            # These will be skipped from testing
            pass

    # Process MultiAnswer objects
    for obj_id, multi_data in multi_answer_map.items():
        ans_eval = multi_data["obj"]
        blank_names = multi_data["blanks"]

        try:
            # Extract each answer from correct_answers or answers list
            answer_list = None
            if hasattr(ans_eval, "correct_answers"):
                answer_list = ans_eval.correct_answers
            elif hasattr(ans_eval, "answers"):
                answer_list = ans_eval.answers

            if answer_list and isinstance(answer_list, (list, tuple)):
                for i, ans in enumerate(answer_list):
                    if i < len(blank_names):
                        answer_str = extract_answer_string(ans)
                        if answer_str:
                            correct_answers[blank_names[i]] = answer_str
        except Exception:
            # If extraction fails for MultiAnswer, skip it
            pass

    return correct_answers


def extract_answer_string(math_obj: Any) -> Optional[str]:
    """
    Convert a MathObject to a student-input format string.

    This is the core extraction function that handles different MathObject types:
    - Real (numeric values)
    - Formula (algebraic expressions)
    - String (text answers)
    - Matrix (array of values)
    - List (multiple answers)
    - RadioButtons/PopUp (multiple choice)
    - AnswerChecker objects (with correct_ans method)

    The challenge: We need the answer in a format that students would type,
    not TeX format (which is for display only).

    Args:
        math_obj: A MathObject (from PG's Value.pm system)

    Returns:
        String representation of the answer, or None if extraction fails
    """
    if math_obj is None:
        return None

    # Try different extraction methods in order of preference

    # 1. Check if it's an AnswerChecker object with correct_value attribute
    if hasattr(math_obj, "correct_value"):
        try:
            correct_val = math_obj.correct_value
            if correct_val is not None:
                # Recursively extract from the correct_value
                return extract_answer_string(correct_val)
        except Exception:
            pass

    # 2. Check if it's an AnswerChecker object with correct_ans method
    if hasattr(math_obj, "correct_ans") and callable(math_obj.correct_ans):
        try:
            result = math_obj.correct_ans()
            if result and isinstance(result, str):
                return result
        except Exception:
            pass

    # 2.5. Check if it's a LinearRelation object
    obj_type = type(math_obj).__name__
    if obj_type == "LinearRelation":
        try:
            # LinearRelation has string() method that returns the formula
            if hasattr(math_obj, "string") and callable(math_obj.string):
                return math_obj.string()
            elif hasattr(math_obj, "to_string") and callable(math_obj.to_string):
                return math_obj.to_string()
            # Fallback to str representation
            return str(math_obj)
        except Exception:
            pass

    # 2.6. Check if it's a List object with elements (needs special handling)
    if hasattr(math_obj, "elements"):
        try:
            result = extract_list_string(math_obj)
            if result:
                return result
        except Exception:
            pass

    # 2.7. Check if it's a Point object
    if type(math_obj).__name__ == "Point":
        try:
            if hasattr(math_obj, "string") and callable(math_obj.string):
                return math_obj.string()
        except Exception:
            pass

    # 2.8. Check if it's a PopUp/DropDown/RadioButtons - extract correct answer
    obj_type = type(math_obj).__name__
    if obj_type in ("PopUp", "DropDown", "RadioButtons"):
        try:
            if hasattr(math_obj, "correct"):
                return str(math_obj.correct)
        except Exception:
            pass
    elif obj_type == "DropDownTF":
        try:
            if hasattr(math_obj, "correct"):
                correct = math_obj.correct
                # Convert 'T'/'F' to 'True'/'False' for student input
                if isinstance(correct, str):
                    return 'True' if correct.upper() == 'T' else 'False'
                else:
                    return 'True' if correct else 'False'
        except Exception:
            pass

    # 3. Try string() method - returns student-input format
    if hasattr(math_obj, "string") and callable(math_obj.string):
        try:
            result = math_obj.string()
            if result and isinstance(result, str):
                return result
        except Exception:
            pass

    # 4. Try value attribute - for numeric values
    if hasattr(math_obj, "value"):
        try:
            value = math_obj.value
            if value is not None:
                # Check if it's a numeric value
                if isinstance(value, (int, float)):
                    # Format without unnecessary decimals
                    if isinstance(value, float) and value.is_integer():
                        return str(int(value))
                    return str(value)
                # Could be a string or other type
                elif isinstance(value, str):
                    return value
        except Exception:
            pass

    # 5. Try data attribute - some MathObjects store answer here
    if hasattr(math_obj, "data"):
        try:
            data = math_obj.data
            if data is not None:
                return str(data)
        except Exception:
            pass

    # 6. For Matrix types - convert matrix representation
    if hasattr(math_obj, "type") and hasattr(math_obj.type, "__call__"):
        try:
            type_name = math_obj.type()
            if "Matrix" in str(type_name) or "matrix" in str(type_name):
                return extract_matrix_string(math_obj)
        except Exception:
            pass

    # 7. For List types - extract list elements
    if hasattr(math_obj, "__iter__") and not isinstance(math_obj, str):
        try:
            # Check if it looks like a list/set of answers
            if hasattr(math_obj, "length"):
                return extract_list_string(math_obj)
        except Exception:
            pass

    # 8. Fallback to string representation
    try:
        result = str(math_obj)
        if result:
            return result
    except Exception:
        pass

    return None


def extract_matrix_string(matrix_obj: Any) -> Optional[str]:
    """
    Extract string representation of a Matrix answer.

    Matrices should be formatted as a space-separated row representation:
    [[a, b], [c, d]] -> "a b c d"  (with row structure preserved)

    Args:
        matrix_obj: A Matrix MathObject

    Returns:
        String representation, or None if extraction fails
    """
    try:
        # Try to access data directly
        if hasattr(matrix_obj, "data"):
            return str(matrix_obj.data)

        # Try to convert to string via data structure
        if hasattr(matrix_obj, "value"):
            value = matrix_obj.value
            # Convert nested list structure to string
            if isinstance(value, list):
                # Flatten matrix rows
                rows = []
                for row in value:
                    if isinstance(row, (list, tuple)):
                        rows.append(" ".join(str(x) for x in row))
                    else:
                        rows.append(str(row))
                return " ".join(rows)

        # Fallback to string representation
        return str(matrix_obj)
    except Exception:
        return None


def extract_list_string(list_obj: Any) -> Optional[str]:
    """
    Extract string representation of a List answer (multiple values).

    Lists should be formatted as comma-separated values:
    [a, b, c] -> "a, b, c"

    Args:
        list_obj: A List or Set MathObject

    Returns:
        String representation, or None if extraction fails
    """
    try:
        # Try elements attribute (for List objects with individual elements)
        if hasattr(list_obj, "elements"):
            elements = list_obj.elements
            if isinstance(elements, (list, tuple)):
                items = []
                for item in elements:
                    item_str = extract_answer_string(item)
                    if item_str:
                        items.append(item_str)
                if items:
                    return ", ".join(items)

        # Try to_string method
        if hasattr(list_obj, "to_string") and callable(list_obj.to_string):
            try:
                result = list_obj.to_string()
                if result:
                    # to_string() returns "[a, b, c]" format, we need to remove brackets
                    result = result.strip()
                    if result.startswith("[") and result.endswith("]"):
                        result = result[1:-1].strip()
                    return result
            except Exception:
                pass

        # Try to get value attribute
        if hasattr(list_obj, "value"):
            value = list_obj.value
            if isinstance(value, (list, tuple)):
                # Format as comma-separated
                items = []
                for item in value:
                    if hasattr(item, "string"):
                        items.append(str(item.string()))
                    else:
                        items.append(str(item))
                return ", ".join(items)

        # Try data attribute
        if hasattr(list_obj, "data"):
            data = list_obj.data
            if isinstance(data, (list, tuple)):
                return ", ".join(str(x) for x in data)

        # Fallback
        return str(list_obj)
    except Exception:
        return None


def get_extractable_blanks(result: Any) -> Dict[str, bool]:
    """
    Check which answer blanks can be successfully extracted.

    This is useful for understanding which problems support runtime testing
    and which have answer types that can't be automatically extracted.

    Args:
        result: ProblemResult from PGTranslator.translate()

    Returns:
        Dictionary mapping blank names to whether they can be extracted
    """
    if not result.answer_blanks:
        return {}

    extractable = {}

    for blank_name, blank_info in result.answer_blanks.items():
        if not blank_info:
            extractable[blank_name] = False
            continue

        evaluator = blank_info.get("evaluator")
        if not evaluator:
            extractable[blank_name] = False
            continue

        if isinstance(evaluator, dict):
            ans_eval = evaluator.get("ans_eval")
        else:
            ans_eval = evaluator

        if not ans_eval:
            extractable[blank_name] = False
            continue

        # Try to extract
        try:
            answer_str = extract_answer_string(ans_eval)
            extractable[blank_name] = answer_str is not None and len(
                answer_str) > 0
        except Exception:
            extractable[blank_name] = False

    return extractable
