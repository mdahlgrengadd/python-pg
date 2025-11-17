"""
Answer checkers for MathObjects.

Provides answer checking functionality for Formula objects in pg_math.
Ported from pg.mathobjects for Perl parity migration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field
import random
import sympy as sp


class AnswerChecker(BaseModel, ABC):
    """
    Base class for answer checkers.

    Pydantic-based answer checker with validation and type safety.
    Supports method chaining and post-filtering.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    correct_value: Any = Field(description="The correct answer")
    options: dict = Field(default_factory=dict, description="Checker options dict")
    post_filter: Optional[Callable] = Field(default=None, description="Post-processing filter function")

    @abstractmethod
    def check(self, student_answer: str) -> Dict[str, Any]:
        """
        Check a student answer.

        Args:
            student_answer: Student's answer as string

        Returns:
            Dictionary with 'score' (0-1) and optional 'message'
        """
        raise NotImplementedError("Subclass must implement check()")

    def withPostFilter(self, filter_function: Callable) -> AnswerChecker:
        """
        Add post-processing filter (stub implementation).

        In full implementation, this would apply a filter function after
        answer checking to provide custom hints, modify scores, etc.

        Args:
            filter_function: Filter to apply (e.g., AnswerHints result)

        Returns:
            self (for method chaining)
        """
        # Stub: just store the filter but don't use it
        self.post_filter = filter_function
        return self


class FormulaAnswerChecker(AnswerChecker):
    """
    Answer checker for Formula objects.

    Compares formulas by testing them at multiple points.
    Pydantic-based with field validators for option extraction.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    num_points: int = Field(default=5, ge=1, description="Number of test points for formula comparison")
    tolerance: float = Field(default=0.01, gt=0, description="Tolerance for formula comparison")

    def __init__(self, correct_value, **options):
        """
        Create a FormulaAnswerChecker.

        Args:
            correct_value: The correct Formula
            **options: Checker options (num_points, tolerance)
        """
        # Extract num_points and tolerance from options if not provided directly
        num_points = options.pop('num_points', None)
        tolerance = options.pop('tolerance', None)

        super().__init__(
            correct_value=correct_value,
            options=options,
            num_points=num_points or 5,
            tolerance=tolerance or 0.01
        )

    def check(self, student_answer: str) -> Dict[str, Any]:
        """
        Check if student answer matches correct answer.

        Tests the formulas at multiple random points to see if they
        produce the same values within tolerance.

        Args:
            student_answer: Student's answer (string)

        Returns:
            dict with 'score' and 'correct' keys
        """
        from .formula import Formula

        # Parse student answer to Formula
        try:
            # Use the correct formula's context to ensure same flags/variables
            student_formula = Formula(
                student_answer, self.correct_value.variables, self.correct_value.context)
        except Exception as e:
            return {
                'score': 0.0,
                'correct': False,
                'message': f'Error parsing answer: {e}'
            }

        # Get variables from correct answer
        # Check if _sympy_expr exists and is not None
        if self.correct_value._sympy_expr is None:
            # Fall back to string comparison if SymPy expression is not available
            correct_str = str(self.correct_value).strip()
            student_str = str(student_formula).strip()
            is_correct = (correct_str == student_str)
            return {
                'score': 1.0 if is_correct else 0.0,
                'correct': is_correct,
                'message': '' if is_correct else 'Answer does not match'
            }
        
        if student_formula._sympy_expr is None:
            # Student formula couldn't be parsed, compare as strings
            correct_str = str(self.correct_value).strip()
            student_str = str(student_formula).strip()
            is_correct = (correct_str == student_str)
            return {
                'score': 1.0 if is_correct else 0.0,
                'correct': is_correct,
                'message': '' if is_correct else 'Could not parse student answer'
            }
        
        correct_vars = sorted(
            [str(s) for s in self.correct_value._sympy_expr.free_symbols])
        student_vars = sorted(
            [str(s) for s in student_formula._sympy_expr.free_symbols])

        # Check that variables match
        if correct_vars != student_vars:
            return {
                'score': 0.0,
                'correct': False,
                'message': f'Formula uses different variables. Expected: {correct_vars}, got: {student_vars}'
            }

        # Use Formula's compare() method for full test point evaluation with domain checking
        # This matches Perl's behavior (lib/Value/Formula.pm::cmp_compare)
        try:
            # Set number of test points on student formula
            student_formula._num_test_points = self.num_points
            
            # Use Formula.compare() which handles:
            # - Symbolic comparison first (fast path)
            # - Test point generation
            # - Domain mismatch detection
            # - Tolerance-based comparison
            is_equal = self.correct_value.compare(student_formula, self.tolerance)
            
            # Check for domain mismatches
            if hasattr(self.correct_value, 'domain_mismatch') and self.correct_value.domain_mismatch:
                return {
                    'score': 0.0,
                    'correct': False,
                    'message': 'The formulas have different domains (one is undefined where the other is defined)'
                }
            
            if is_equal:
                return {
                    'score': 1.0,
                    'correct': True,
                    'message': 'Correct!'
                }
            else:
                return {
                    'score': 0.0,
                    'correct': False,
                    'message': 'The formulas are not equivalent'
                }
                
        except Exception as e:
            # Fallback to simplified test point evaluation if compare() fails
            # Test at multiple random points
            for _ in range(self.num_points):
                # Generate random test point
                test_point = {}
                for var in correct_vars:
                    test_point[var] = random.uniform(-5, 5)

                try:
                    # Evaluate both formulas
                    correct_value = self.correct_value.eval(**test_point)
                    student_value = student_formula.eval(**test_point)

                    # Compare values - convert to float
                    correct_float = float(correct_value)
                    student_float = float(student_value)

                    diff = abs(correct_float - student_float)

                    # Check if difference is within tolerance
                    if diff > self.tolerance:
                        return {
                            'score': 0.0,
                            'correct': False,
                            'message': f'Formulas differ at test point {test_point}'
                        }
                except Exception as e:
                    # Evaluation error at this point - might be domain issue
                    # Check if both formulas have the same domain issue
                    try:
                        # Try to evaluate correct formula
                        self.correct_value.eval(**test_point)
                        # Correct formula works, student doesn't - domain mismatch
                        return {
                            'score': 0.0,
                            'correct': False,
                            'message': 'The formulas have different domains (one is undefined where the other is defined)'
                        }
                    except Exception:
                        # Both fail - might be OK, continue to next point
                        pass

            # All test points passed
            return {
                'score': 1.0,
                'correct': True,
                'message': 'Correct!'
            }


class RealAnswerChecker(AnswerChecker):
    """
    Answer checker for Real numbers.

    Compares real numbers with tolerance.
    Pydantic-based with context-aware default extraction via model_post_init.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    tolerance: Optional[float] = Field(default=None, gt=0, description="Tolerance for real number comparison")
    tol_type: Optional[str] = Field(default=None, description="Tolerance type: 'relative', 'absolute', or 'sigfigs'")

    def __init__(self, correct_value, **options):
        """
        Create a RealAnswerChecker.

        Args:
            correct_value: The correct Real number
            **options: Checker options (tolerance, tolType)
        """
        # Extract tolerance and tol_type from options
        tolerance = options.pop('tolerance', None)
        tol_type = options.pop('tolType', None)

        super().__init__(
            correct_value=correct_value,
            options=options,
            tolerance=tolerance,
            tol_type=tol_type
        )

    def model_post_init(self, __context: Any) -> None:
        """Extract tolerance and tol_type from context if not provided."""
        # Extract tolerance from context if not provided
        if self.tolerance is None:
            if hasattr(self.correct_value, 'context') and self.correct_value.context is not None:
                ctx_tolerance = self.correct_value.context.flags.get('tolerance')
                if ctx_tolerance is not None:
                    self.tolerance = ctx_tolerance
        if self.tolerance is None:
            self.tolerance = 0.001

        # Extract tol_type from context if not provided
        if self.tol_type is None:
            if hasattr(self.correct_value, 'context') and self.correct_value.context is not None:
                ctx_tol_type = self.correct_value.context.flags.get('tolType')
                if ctx_tol_type is not None:
                    self.tol_type = ctx_tol_type
        if self.tol_type is None:
            self.tol_type = 'relative'

    def __call__(self, student_answer: str) -> Dict[str, Any]:
        """Allow checker to be called as a function."""
        return self.check(student_answer)

    def check(self, student_answer: str) -> Dict[str, Any]:
        """
        Check if student answer matches correct answer.

        Args:
            student_answer: Student's answer (string)

        Returns:
            dict with 'score', 'correct' keys, and optional 'message'
        """
        from .numeric import Real

        # Try to parse student answer as number
        try:
            if isinstance(student_answer, (int, float)):
                student_value = float(student_answer)
            else:
                student_answer = str(student_answer).strip()
                student_value = float(student_answer)
        except (ValueError, TypeError) as e:
            return {
                'score': 0.0,
                'correct': False,
                'message': f'Invalid number format: {student_answer}'
            }

        # Create a Real from student answer with same context
        student_real = Real(student_value, self.correct_value.context)

        # Compare using Real's equality with tolerance
        if self.correct_value == student_real:
            return {
                'score': 1.0,
                'correct': True
            }
        else:
            return {
                'score': 0.0,
                'correct': False,
                'message': f'Expected {self.correct_value.value}, got {student_value}'
            }


class VectorAnswerChecker(AnswerChecker):
    """
    Answer checker for Vector objects.

    Compares vectors component-wise with tolerance or uses custom checker.
    Pydantic-based with field extraction from options.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    custom_checker: Optional[Callable] = Field(default=None, description="Custom checker function for vectors")
    tolerance: float = Field(default=0.001, gt=0, description="Tolerance for vector component comparison")

    def __init__(self, correct_value, **options):
        """
        Create a VectorAnswerChecker.

        Args:
            correct_value: The correct Vector
            **options: Checker options (tolerance, tolType, checker)
        """
        # Extract custom_checker and tolerance from options
        custom_checker = options.pop('checker', None)
        tolerance = options.pop('tolerance', None)

        super().__init__(
            correct_value=correct_value,
            options=options,
            custom_checker=custom_checker,
            tolerance=tolerance or 0.001
        )

    def __call__(self, student_answer: str) -> Dict[str, Any]:
        """Allow checker to be called as a function."""
        return self.check(student_answer)

    def check(self, student_answer: str) -> Dict[str, Any]:
        """
        Check if student answer matches correct answer.

        Uses Parser/Formula system like Perl (Parser::Formula equivalent via Compute).

        Args:
            student_answer: Student's answer (string or Vector)

        Returns:
            dict with 'score', 'correct' keys, and optional 'message'
        """
        from .geometric import Vector
        from .compute import Compute

        # Parse student answer if it's a string
        if isinstance(student_answer, str):
            student_answer = student_answer.strip()
            
            # Use Compute() to parse (equivalent to Parser::Formula in Perl)
            try:
                # Get context from correct vector
                context = getattr(self.correct_value, 'context', None)
                if context is None:
                    from .context import get_current_context
                    context = get_current_context()
                
                # Parse using Compute (handles both simple vectors and parametric formulas)
                parsed_value = Compute(student_answer, context)
                
                # If it's already a Vector, use it
                if isinstance(parsed_value, Vector):
                    student_vector = parsed_value
                # If it's a Formula that evaluates to a Vector, try to evaluate
                elif hasattr(parsed_value, 'isConstant') and parsed_value.isConstant():
                    try:
                        # Evaluate constant formula - might give us a Vector
                        evaluated = parsed_value.eval()
                        if isinstance(evaluated, Vector):
                            student_vector = evaluated
                        else:
                            # Not a vector - try string comparison
                            return self._compare_strings(student_answer)
                    except Exception:
                        # Evaluation failed - try string comparison
                        return self._compare_strings(student_answer)
                else:
                    # Formula that might represent a parametric vector
                    # For parametric vectors, compare string representations
                    # (Full implementation would evaluate at test points)
                    return self._compare_strings(student_answer)
                    
            except Exception as e:
                # Parsing failed - try string comparison as fallback
                return self._compare_strings(student_answer)
            
            # Use the parsed vector for comparison below
            student_answer = student_vector
        else:
            # Already a Vector object
            student_vector = student_answer

        # If custom checker provided, use it
        if self.custom_checker is not None:
            try:
                # Call custom checker: checker(correct, student, ansHash)
                # ansHash is a stub object for now
                ans_hash = {'correct_ans': self.correct_value}
                result = self.custom_checker(self.correct_value, student_answer, ans_hash)
                # Custom checker returns 0 or 1
                score = float(result) if isinstance(result, (int, float)) else 0.0
                return {
                    'score': score,
                    'correct': score >= 1.0
                }
            except Exception as e:
                return {
                    'score': 0.0,
                    'correct': False,
                    'message': f'Custom checker error: {e}'
                }

        # Default: component-wise comparison
        if not isinstance(student_vector, Vector):
            return {
                'score': 0.0,
                'correct': False,
                'message': 'Student answer is not a Vector'
            }

        # Compare dimensions
        if len(self.correct_value.components) != len(student_vector.components):
            return {
                'score': 0.0,
                'correct': False,
                'message': 'Vector dimensions do not match'
            }

        # Compare component-wise using compare method
        if self.correct_value.compare(student_vector, self.tolerance):
            return {
                'score': 1.0,
                'correct': True
            }
        else:
            return {
                'score': 0.0,
                'correct': False,
                'message': 'Vectors do not match'
            }
    
    def _compare_strings(self, student_answer: str) -> Dict[str, Any]:
        """
        Fallback string comparison for vectors.
        
        Used when parsing fails or for parametric vector formulas.
        """
        correct_str = str(self.correct_value).strip()
        student_str = student_answer.strip()
        
        # Normalize by removing angle brackets
        if correct_str.startswith('<') and correct_str.endswith('>'):
            correct_str = correct_str[1:-1].strip()
        if student_str.startswith('<') and student_str.endswith('>'):
            student_str = student_str[1:-1].strip()
        
        if correct_str == student_str:
            return {
                'score': 1.0,
                'correct': True
            }
        
        return {
            'score': 0.0,
            'correct': False,
            'message': f'Could not parse vector: {student_answer}'
        }
