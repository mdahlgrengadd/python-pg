"""FormulaUpToConstant - Formulas unique up to an arbitrary constant.

This module implements formulas that represent antiderivatives, where
the answer is only unique up to an arbitrary constant (e.g., "+ C").

Students must include a constant in their answer, but can use any
single-letter constant they want.

Reference: lib/pg/macros/parserFormulaUpToConstant.pl in legacy Perl codebase

Example:
    >>> from pg.math import FormulaUpToConstant, get_context
    >>> ctx = get_context('Numeric')
    >>> f = FormulaUpToConstant("x^2/2 + C", context=ctx)
    >>> # Student can answer with any constant
    >>> checker = f.cmp()
    >>> result = checker("x^2/2 + K")  # Acceptable
    >>> result['correct']  # True
"""

from __future__ import annotations

import re
from typing import Any, Dict

from pydantic import ConfigDict, Field, PrivateAttr

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

from .context import Context, get_current_context
from .formula import Formula
from .numeric import Real


class FormulaUpToConstant(Formula):
    """A formula that is only unique up to an arbitrary constant.

    This is used for antiderivatives where the student must include
    a "+ C" (or any other single-letter constant) in their answer.

    Attributes:
        constant (str | None): The name of the arbitrary constant (e.g., "C", "K")

    Reference: parserFormulaUpToConstant.pl
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra='allow',
    )

    # Pydantic fields
    constant: str | None = Field(default=None, description="Name of the arbitrary constant")

    # Private attributes
    _arbitrary_constants: set[str] = PrivateAttr(default_factory=set)
    _private_context: Any = PrivateAttr(default=None)

    def __init__(
        self,
        expression: str | Any,
        variables: list[str] | None = None,
        context: Context | None = None,
    ):
        """Initialize a FormulaUpToConstant.

        Args:
            expression: Mathematical expression (string, sympy expr, or Formula)
            variables: List of variable names (auto-detected if None)
            context: Context to use (creates private copy)

        Notes:
            - If no constant is found in the expression, "+ C" is added automatically
            - The constant must be a single letter (except 'e', 'i', or 'pi')
            - Creates a private context copy to avoid affecting the rest of the problem
        """
        # Create private context copy
        if context is None:
            context = get_current_context()

        private_context = context.copy()

        # Handle different input types
        if isinstance(expression, FormulaUpToConstant):
            # Copy from existing FormulaUpToConstant
            super().__init__(expression.expression, expression.variables, private_context)
            constant_value = expression.constant
            arbitrary_constants = expression._arbitrary_constants.copy()
        elif isinstance(expression, Formula):
            # Convert from Formula
            super().__init__(expression.expression, expression.variables, private_context)
            constant_value = None
            arbitrary_constants = set()
        else:
            # Parse new expression
            super().__init__(expression, variables, private_context)
            constant_value = None
            arbitrary_constants = set()

        # Store private context
        self._private_context = private_context
        self._arbitrary_constants = arbitrary_constants

        # Find the arbitrary constant in the expression
        self.constant = constant_value or self._find_constant()

        # If no constant found, add C automatically
        if self.constant is None and SYMPY_AVAILABLE:
            # Add C to context if not already there
            if self._private_context.variables.get('C') is None:
                self._private_context.variables.add('C')

            # Add C to variables list
            if 'C' not in self.variables:
                self.variables.append('C')

            # Create new formula with + C
            if self._sympy_expr is not None:
                c_sym = sp.Symbol('C')
                new_expr = self._sympy_expr + c_sym
                # Update the sympy expression
                self._sympy_expr = new_expr
            else:
                # If sympy expression is None, append "+C" to string representation
                if hasattr(self, 'expression') and self.expression:
                    self.expression = f"{self.expression} + C"
            self.constant = 'C'
            self._arbitrary_constants.add('C')

        # Verify the formula is linear in the constant
        self._verify_linearity()

    def _find_constant(self) -> str | None:
        """Find the arbitrary constant in the expression.

        Returns:
            Name of the constant variable, or None if not found

        Notes:
            A constant is a single-letter variable that:
            - Is not 'e', 'i', or a known math constant
            - Is not already a defined variable in the base context
        """
        if not SYMPY_AVAILABLE or self._sympy_expr is None:
            return None

        # Get all free symbols from the expression
        symbols = self._sympy_expr.free_symbols

        # Reserved names that can't be constants
        reserved = {'e', 'i', 'pi', 'E', 'I'}

        potential_constants = []
        for sym in symbols:
            name = str(sym)
            if len(name) == 1 and name not in reserved:
                # Check if it's in our arbitrary constants set or not defined yet
                if name in self._arbitrary_constants:
                    potential_constants.append(name)
                elif self._private_context.variables.get(name) is None:
                    # Not defined yet - could be a constant
                    potential_constants.append(name)

        if len(potential_constants) > 1:
            raise ValueError(
                f"Formula has multiple potential arbitrary constants: "
                f"{', '.join(potential_constants)}. Use only one constant."
            )

        if len(potential_constants) == 1:
            # Mark it as arbitrary constant
            const_name = potential_constants[0]
            if self._private_context.variables.get(const_name) is None:
                self._private_context.variables.add(const_name)
            self._arbitrary_constants.add(const_name)
            return const_name

        return None

    def _verify_linearity(self) -> None:
        """Verify that the formula is linear in the arbitrary constant.

        Raises:
            ValueError: If the formula is not linear in the constant
        """
        if self.constant is None or not SYMPY_AVAILABLE or self._sympy_expr is None:
            return

        # Differentiate with respect to the constant
        const_sym = sp.Symbol(self.constant)
        derivative_expr = sp.diff(self._sympy_expr, const_sym)

        # Check if derivative has any free symbols
        # For linearity, the derivative with respect to C should be constant (no variables)
        if len(derivative_expr.free_symbols) > 0:
            raise ValueError(
                f"Formula is not linear in the arbitrary constant '{self.constant}'. "
                f"The derivative with respect to {self.constant} must be constant, "
                f"but got: {derivative_expr}"
            )

    def compare_up_to_constant(
        self,
        other: Any,
        tolerance: float | None = None,
    ) -> tuple[bool, str | None]:
        """Compare this formula with another, accounting for arbitrary constants.

        Args:
            other: The value to compare with
            tolerance: Comparison tolerance (uses context default if None)

        Returns:
            Tuple of (equal, error_message)
            - equal: True if equivalent up to constant
            - error_message: Helpful message if not equal, None if equal

        Notes:
            Two FormulaUpToConstant objects are equal if they differ only
            by a constant term.
        """
        # Check if other is a FormulaUpToConstant or Formula
        if isinstance(other, (Real, int, float)):
            return False, "Your answer should include an arbitrary constant (like +C)"

        if isinstance(other, Formula) and not isinstance(other, FormulaUpToConstant):
            return False, "Your answer should include an arbitrary constant (like +C)"

        if not isinstance(other, FormulaUpToConstant):
            return False, "Your answer should include an arbitrary constant (like +C)"

        # Check if other has a constant
        if other.constant is None:
            return False, "Your answer should include an arbitrary constant (like +C)"

        if not SYMPY_AVAILABLE:
            return False, "Cannot compare formulas without SymPy"

        # If constants are different, substitute to match
        other_expr = other._sympy_expr
        if other.constant != self.constant:
            # Substitute other's constant with ours
            other_sym = sp.Symbol(other.constant)
            self_sym = sp.Symbol(self.constant)
            other_expr = other_expr.subs(other_sym, self_sym)

        # Now compare: self.expr and other_expr should differ by at most a constant
        # This means: self.expr - other_expr should be a constant
        diff = sp.simplify(self._sympy_expr - other_expr)

        # Check if diff is constant (no free symbols except the arbitrary constant)
        diff_symbols = diff.free_symbols
        const_sym = sp.Symbol(self.constant)

        # Remove the arbitrary constant from free symbols
        diff_symbols_without_const = diff_symbols - {const_sym}

        if len(diff_symbols_without_const) > 0:
            return False, "Your answer is not equivalent to the correct answer"

        # Formulas are equivalent up to constant!
        return True, None

    def compare(
        self,
        other: Any,
        tolerance: float | None = None,
    ) -> tuple[bool, str | None]:
        """Alias for compare_up_to_constant for API compatibility.

        Args:
            other: The value to compare with
            tolerance: Comparison tolerance (uses context default if None)

        Returns:
            Tuple of (equal, error_message)
        """
        return self.compare_up_to_constant(other, tolerance)

    def remove_constant(self) -> Formula:
        """Remove the arbitrary constant and return a regular Formula.

        Returns:
            Formula with the constant set to zero

        Example:
            >>> f = FormulaUpToConstant("x^2/2 + C")
            >>> g = f.remove_constant()  # Returns Formula("x^2/2")
        """
        if self.constant is None or not SYMPY_AVAILABLE:
            return Formula(self.expression, self.variables, self._private_context)

        # Substitute constant with 0
        const_sym = sp.Symbol(self.constant)
        expr_without_const = self._sympy_expr.subs(const_sym, 0)

        # Simplify and return as Formula
        expr_simplified = sp.simplify(expr_without_const)
        return Formula(expr_simplified, self.variables, self._private_context)

    def diff(self, var: str) -> Formula:
        """Differentiate with respect to a variable.

        Args:
            var: Variable name to differentiate with respect to

        Returns:
            Formula (not FormulaUpToConstant) representing the derivative

        Notes:
            Returns a regular Formula because derivatives don't need the constant
        """
        return self.remove_constant().diff(var)

    def D(self, var: str) -> Formula:
        """Alias for diff() for API compatibility with Perl MathObjects.

        Args:
            var: Variable name to differentiate with respect to

        Returns:
            Formula (not FormulaUpToConstant) representing the derivative
        """
        return self.diff(var)

    def cmp(self, **options):
        """Create an answer checker for this FormulaUpToConstant.

        Args:
            **options: Options for the answer checker
                showHints (bool): Show helpful hints to students (default: True)
                showLinearityHints (bool): Show linearity hints (default: True)
                tolerance (float): Comparison tolerance

        Returns:
            Callable that checks student answers

        Reference: parserFormulaUpToConstant.pl cmp() method
        """
        show_hints = options.get('showHints', True)
        show_linearity_hints = options.get('showLinearityHints', True)
        tolerance = options.get('tolerance', None)

        def checker(student_answer: str) -> Dict[str, Any]:
            """Check student answer against this FormulaUpToConstant.

            Args:
                student_answer: Student's answer string

            Returns:
                Dictionary with:
                    - correct: True if answer is correct
                    - message: Feedback message for student
                    - student_value: Parsed student answer
            """
            from .compute import Compute

            result = {
                'correct': False,
                'message': '',
                'student_value': None,
            }

            try:
                # Parse student answer using Compute
                student = Compute(student_answer, self._private_context)
                result['student_value'] = student

                # Try to convert to FormulaUpToConstant if it's a Formula
                if isinstance(student, Formula) and not isinstance(student, FormulaUpToConstant):
                    # Check if it has a potential constant
                    # Get the variables from the correct answer (excluding the constant)
                    correct_vars = set(
                        self._private_context.variables.list()) - {self.constant}

                    # Check student's symbols
                    if SYMPY_AVAILABLE and student._sympy_expr:
                        symbols = student._sympy_expr.free_symbols
                        potential_constants = [
                            str(s) for s in symbols
                            if (len(str(s)) == 1 and
                                str(s) not in {'e', 'i', 'E', 'I'} and
                                # Exclude problem variables
                                str(s) not in correct_vars)
                        ]

                        if len(potential_constants) == 0:
                            result['message'] = (
                                "Your answer should include an arbitrary constant (like +C). "
                                "Note: there is always more than one possibility."
                            ) if show_hints else ""
                            return result
                        elif len(potential_constants) == 1:
                            # Convert to FormulaUpToConstant
                            try:
                                student = FormulaUpToConstant(
                                    student, context=self._private_context)
                            except ValueError as e:
                                if show_linearity_hints:
                                    result['message'] = str(e)
                                return result

                # Compare
                equal, error_msg = self.compare_up_to_constant(
                    student, tolerance)

                if equal:
                    result['correct'] = True
                    result['message'] = ""
                elif show_hints and error_msg:
                    result['message'] = error_msg

            except Exception as e:
                result['message'] = f"Error parsing answer: {str(e)}" if show_hints else ""

            return result

        return checker

    def TeX(self) -> str:
        """Generate LaTeX/TeX representation.

        Returns:
            TeX string representation of the formula
        """
        if not SYMPY_AVAILABLE or self._sympy_expr is None:
            return self.to_string()

        # Use sympy's latex function
        import sympy as sp
        return sp.latex(self._sympy_expr)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"FormulaUpToConstant('{self.to_string()}', constant='{self.constant}')"
