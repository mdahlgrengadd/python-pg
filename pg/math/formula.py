"""
Formula type for the WeBWorK PG MathObjects system.

Formula wraps an AST (Abstract Syntax Tree) for deferred evaluation, providing:
- Variable substitution
- Differentiation (using SymPy)
- Simplification/reduction
- Multiple evaluation modes
- Multiple output formats
- Test point evaluation for comparison
- Python function generation
- Domain checking

Reference: lib/Value/Formula.pm (1,156 lines) in legacy Perl codebase

NEW FEATURES FOR PARITY:
- create_random_points(): Generate random test points for formula testing
- create_point_values(): Evaluate formula at test points with error handling
- python_function(): Convert formula to executable Python function
- Domain checking with undefined point handling
- Adaptive parameter support
- Full comparison with test point evaluation
- cmp(): Built-in answer checker
"""

from __future__ import annotations
from .value import MathValue, ToleranceMode, TypePrecedence

import random
import types
from typing import Any, Callable

_SYMPY_TRANSFORMATIONS: tuple = ()
try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr,
        implicit_multiplication_application,
        standard_transformations,
    )

    _SYMPY_TRANSFORMATIONS = (
        standard_transformations + (implicit_multiplication_application,)
    )
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


class CMPWrapper:
    """
    Wrapper for .cmp that works as both property access and method call.

    Handles Perl idiom where ->cmp can be used without parentheses and
    chained with ->withPostFilter.
    """

    def __init__(self, formula, **default_options):
        self.formula = formula
        self.default_options = default_options

    def __call__(self, **options):
        """Call as cmp() to get answer checker."""
        from .answer_checker import FormulaAnswerChecker
        merged_options = {**self.default_options, **options}
        return FormulaAnswerChecker(self.formula, **merged_options)

    def withPostFilter(self, filter_function):
        """Chain with withPostFilter like Perl ->cmp->withPostFilter."""
        from .answer_checker import FormulaAnswerChecker
        checker = FormulaAnswerChecker(self.formula, **self.default_options)
        return checker.withPostFilter(filter_function)


class Formula(MathValue):
    """
    Formula represents a mathematical expression with deferred evaluation.

    Unlike numeric types (Real, Complex) which have immediate values, Formula
    stores an expression that can be:
    - Evaluated with variable bindings
    - Differentiated with respect to variables
    - Simplified/reduced
    - Substituted with other values

    Examples:
        >>> f = Formula("x^2 + 2*x + 1")
        >>> f.eval(x=3)  # Returns Real(16)
        >>> f.diff("x")  # Returns Formula("2*x + 2")
        >>> f.substitute("x", Real(5))  # Returns Formula("36")
    """

    type_precedence = TypePrecedence.FORMULA

    def __init__(
        self,
        expression: str | Any,
        variables: list[str] | None = None,
        context: Any | None = None,
        test_points: list[list[float]] | None = None,
        num_test_points: int = 5,
        limits: dict[str, tuple[float, float]] | None = None,
    ):
        """
        Create a Formula from an expression.

        Args:
            expression: Mathematical expression (string or AST node or SymPy expr)
            variables: List of variable names in the expression
            context: Mathematical context (for parsing and evaluation)
            test_points: Pre-specified test points for comparison (optional)
            num_test_points: Number of random test points to generate (default: 5)
            limits: Variable limits for random test point generation {var: (min, max)}
        """
        self.expression = expression
        # Extract variables from expression if not provided
        if variables is None or (isinstance(variables, list) and len(variables) == 0):
            if isinstance(expression, str):
                # Extract variables from the expression string
                import re
                # Common function names to exclude
                function_names = {'sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'asin', 'acos', 'atan', 
                                 'sinh', 'cosh', 'tanh', 'log', 'ln', 'exp', 'sqrt', 'abs', 'sgn', 
                                 'step', 'fact', 'pi', 'e', 'E', 'PI'}
                # Find all word-like tokens
                var_pattern = r'\b([a-zA-Z][a-zA-Z0-9_]*)\b'
                found_vars = set(re.findall(var_pattern, expression))
                # Filter out function names and numbers
                extracted_vars = [v for v in found_vars if v not in function_names and not v.replace('_', '').isdigit()]
                if extracted_vars:
                    variables = extracted_vars
                elif context is not None:
                    # Fallback to context variables
                    variables = context.variables.list()
                else:
                    variables = []
        self.variables = variables or []
        self.context = context

        # Test point configuration (for formula comparison)
        self._test_points = test_points
        self._test_values = None
        self._num_test_points = num_test_points
        self._limits = limits or {}

        # Cached Python function for evaluation
        self._python_func = None

        # Domain checking flags
        self.domain_mismatch = False
        self.check_undefined_points = False
        self.max_undefined = num_test_points

        # Check for assignment expressions if context has assignments enabled
        # This should happen before SymPy parsing
        if isinstance(expression, str) and context is not None:
            if hasattr(context, 'has_assignment_operator') and context.has_assignment_operator():
                if '=' in expression:
                    # Try to parse as assignment
                    from pg.math.compute import _parse_assignment
                    assignment_obj = _parse_assignment(expression, context)
                    if assignment_obj is not None:
                        # This is an assignment - copy assignment attributes
                        self._assignment_value = assignment_obj._assignment_value
                        self._is_assignment = True
                        # Continue with normal Formula initialization for the expression string
                        # The assignment_value will be used for comparison
        
        # If expression is a string and SymPy is available, parse it
        if isinstance(expression, str) and SYMPY_AVAILABLE:
            try:
                # Preprocess: convert Perl ^ operator to Python ** operator
                # Only convert ^ when it appears to be used for exponentiation:
                # After a number, variable, or closing paren/bracket: 2^3, x^2, (x+1)^2, [a]^2
                import re
                processed_expr = re.sub(r'([0-9a-zA-Z_\)\]])\^', r'\1**', expression)

                # Build local_dict with variables as symbols and constants
                local_dict = {var: sp.Symbol(var) for var in self.variables}
                # Add mathematical constants
                local_dict['e'] = sp.E
                local_dict['pi'] = sp.pi
                local_dict['E'] = sp.E
                local_dict['PI'] = sp.pi
                self._sympy_expr = parse_expr(
                    processed_expr,
                    transformations=_SYMPY_TRANSFORMATIONS,
                    local_dict=local_dict,
                )
            except Exception:
                # Fallback: store as string
                self._sympy_expr = None
        else:
            self._sympy_expr = expression if SYMPY_AVAILABLE and isinstance(
                expression, sp.Expr) else None

        # Validate polynomial form if context requires it
        if self.context is not None and hasattr(self.context, 'flags'):
            self._validate_polynomial()

    def _validate_polynomial(self):
        """Validate polynomial form if required by context."""
        # Check for LimitedPolynomial validation
        if self.context.flags.get('limitedPolynomial'):
            # Import here to avoid circular dependency
            from .limited_polynomial import validate_polynomial_formula

            is_valid, error = validate_polynomial_formula(self)
            if not is_valid:
                raise ValueError(error)

        # Check for PolynomialFactors validation
        if self.context.flags.get('polynomialFactors'):
            # Import here to avoid circular dependency
            from .polynomial_factors import validate_factored_polynomial

            is_valid, error = validate_factored_polynomial(self)
            if not is_valid:
                raise ValueError(error)

    def eval(
        self, **bindings: float | MathValue
    ) -> MathValue:
        """
        Evaluate the formula with variable bindings.

        Args:
            **bindings: Variable name -> value mappings

        Returns:
            Result as MathValue (Real, Complex, etc.)

        Example:
            >>> f = Formula("x^2 + 1")
            >>> f.eval(x=3)
            Real(10)
        """
        sympy_failed_message: str | None = None

        if self._sympy_expr is not None:
            sympy_bindings = {}
            for var, value in bindings.items():
                if isinstance(value, MathValue):
                    sympy_bindings[sp.Symbol(var)] = value.to_python()
                elif isinstance(value, str):
                    # Parse string values as sympy expressions
                    try:
                        # Handle implicit multiplication like '2pi' -> '2*pi'
                        # Also convert ^ to ** for exponentiation (only when used as exponent)
                        import re
                        processed_value = re.sub(r'([0-9a-zA-Z_\)\]])\^', r'\1**', value)
                        parsed_value = parse_expr(
                            processed_value,
                            transformations=_SYMPY_TRANSFORMATIONS,
                            local_dict={'pi': sp.pi, 'e': sp.E,
                                        'E': sp.E, 'PI': sp.pi}
                        )
                        sympy_bindings[sp.Symbol(var)] = parsed_value
                    except:
                        # If parsing fails, use the original value
                        sympy_bindings[sp.Symbol(var)] = value
                else:
                    sympy_bindings[sp.Symbol(var)] = value

            try:
                result = self._sympy_expr.subs(sympy_bindings)
            except RecursionError:
                raise ValueError(
                    f"SymPy recursion error during substitution of {self.to_string()}"
                )

            try:
                numeric_value = float(result)
                from .numeric import Real

                return Real(numeric_value)
            except (TypeError, ValueError, RecursionError):
                sympy_failed_message = (
                    f"Formula evaluation resulted in symbolic expression: {result}"
                )

        expr_source = (
            self.expression if isinstance(
                self.expression, str) else str(self.expression)
        )

        try:
            from pg.parser.parser import Parser
            from pg.parser.visitors import EvalVisitor

            # Use the formula's context if available, otherwise try to get the current context
            eval_context = self.context
            if eval_context is None:
                try:
                    from pg.math.context import get_current_context
                    eval_context = get_current_context()
                except (ImportError, RuntimeError):
                    # No current context available
                    pass

            parser = Parser(eval_context)
            ast = parser.parse(expr_source)

            eval_bindings = {}
            for var, value in bindings.items():
                if isinstance(value, MathValue):
                    eval_bindings[var] = value.to_python()
                else:
                    eval_bindings[var] = value

            visitor = EvalVisitor(eval_bindings, eval_context)
            result = ast.accept(visitor)

            return MathValue.from_python(result)
        except ImportError as parser_exc:
            if sympy_failed_message:
                raise ValueError(sympy_failed_message) from parser_exc
            raise RuntimeError(
                "Cannot evaluate Formula: neither SymPy nor pg_parser available"
            ) from parser_exc
        except Exception as parser_exc:
            if sympy_failed_message:
                raise ValueError(sympy_failed_message) from parser_exc
            raise

    @property
    def reduce(self) -> Formula:
        """
        Simplify/reduce the formula.

        In Perl this is called as ->reduce (no parens), so we make it a property.

        Returns:
            Simplified Formula

        Example:
            >>> f = Formula("x + x")
            >>> f.reduce
            Formula("2*x")
        """
        if self._sympy_expr is not None:
            simplified = sp.simplify(self._sympy_expr)
            return Formula(simplified, self.variables, self.context)
        else:
            # No simplification without SymPy
            return self

    def substitute(self, **substitutions) -> 'Formula':
        """
        Substitute expressions for variables.

        Args:
            **substitutions: Variable substitutions (e.g., x='2*y', y='t+1')

        Returns:
            New Formula with substitutions applied

        Example:
            >>> f = Formula("x^2 + y", ['x', 'y'])
            >>> f.substitute(x='2*t', y='t+1')
            Formula("4*t^2 + t + 1")
        """
        if self._sympy_expr is not None:
            subs_dict = {}
            for var, expr in substitutions.items():
                # Parse substitution expression
                if isinstance(expr, str):
                    # Parse using Compute() to handle string expressions
                    from .compute import Compute
                    sub_formula = Compute(expr, self.context)
                    subs_dict[sp.Symbol(var)] = sub_formula._sympy_expr
                elif isinstance(expr, Formula):
                    subs_dict[sp.Symbol(var)] = expr._sympy_expr
                elif isinstance(expr, (int, float)):
                    subs_dict[sp.Symbol(var)] = expr
                else:
                    # Try to get python value
                    from .math_value import MathValue
                    if isinstance(expr, MathValue):
                        python_value = expr.to_python()
                        subs_dict[sp.Symbol(var)] = python_value
                    else:
                        subs_dict[sp.Symbol(var)] = expr

            substituted = self._sympy_expr.subs(subs_dict)

            # Update variable list - remove substituted vars, add new vars
            new_vars = [v for v in self.variables if v not in substitutions]
            # Add variables from substituted expressions
            for expr in substitutions.values():
                if isinstance(expr, Formula):
                    new_vars.extend(
                        v for v in expr.variables if v not in new_vars)

            return Formula(substituted, new_vars, self.context)
        else:
            # Fallback: can't substitute without SymPy
            raise NotImplementedError("Substitution requires SymPy")

    def diff(self, var: str) -> Formula:
        """
        Differentiate with respect to a variable.

        Args:
            var: Variable to differentiate with respect to

        Returns:
            Derivative as Formula

        Example:
            >>> f = Formula("x^2")
            >>> f.diff("x")
            Formula("2*x")
        """
        if not SYMPY_AVAILABLE:
            raise RuntimeError("Differentiation requires SymPy")

        if self._sympy_expr is not None:
            derivative = sp.diff(self._sympy_expr, sp.Symbol(var))
            return Formula(derivative, self.variables, self.context)
        else:
            raise RuntimeError("Cannot differentiate: expression not parsed")

    def D(self, var: str) -> 'Formula':
        """Alias for diff() for Perl API compatibility.

        Args:
            var: Variable to differentiate with respect to

        Returns:
            Derivative as Formula
        """
        return self.diff(var)

    def integrate(self, var: str) -> Formula:
        """
        Integrate with respect to a variable.

        Args:
            var: Variable to integrate with respect to

        Returns:
            Integral as Formula

        Example:
            >>> f = Formula("2*x")
            >>> f.integrate("x")
            Formula("x^2")
        """
        if not SYMPY_AVAILABLE:
            raise RuntimeError("Integration requires SymPy")

        if self._sympy_expr is not None:
            integral = sp.integrate(self._sympy_expr, sp.Symbol(var))
            return Formula(integral, self.variables, self.context)
        else:
            raise RuntimeError("Cannot integrate: expression not parsed")

    # Test point evaluation (NEW FOR PARITY)

    def create_random_points(
        self,
        num_points: int | None = None,
        include: list[list[float]] | None = None,
        no_errors: bool = False,
    ) -> tuple[list[list[float]], list[MathValue | None], bool]:
        """
        Create random test points for formula comparison.

        Generates random points in variable domains and evaluates formula.
        Ensures points where formula is defined (unless check_undefined_points).

        Args:
            num_points: Number of points to generate (default: self._num_test_points)
            include: Additional points to include
            no_errors: If True, return error flag instead of raising

        Returns:
            Tuple of (points, values, has_error):
            - points: List of coordinate lists [[x1, y1], [x2, y2], ...]
            - values: Evaluated values (None for undefined points)
            - has_error: True if couldn't generate enough valid points

        Reference: lib/Value/Formula.pm::createRandomPoints (lines 338-406)
        """
        if num_points is None:
            num_points = self._num_test_points

        if num_points < 1:
            num_points = 1

        points = []
        values = []
        num_undefined = 0

        # Include pre-specified points
        if include:
            points.extend(include)
            for point in include:
                try:
                    val = self.eval(
                        **{var: point[i] for i, var in enumerate(self.variables)})
                    values.append(val)
                except (ValueError, ZeroDivisionError, ArithmeticError):
                    values.append(None)
                    if self.check_undefined_points:
                        num_undefined += 1

        # Generate random points
        attempts = 0
        max_attempts = num_points * 10

        while len(points) - num_undefined < num_points and attempts < max_attempts:
            # Generate random point
            point = []
            for var in self.variables:
                if var in self._limits:
                    low, high = self._limits[var]
                else:
                    low, high = -10.0, 10.0  # Default range

                point.append(random.uniform(low, high))

            # Try to evaluate at this point
            try:
                bindings = {var: point[i]
                            for i, var in enumerate(self.variables)}
                val = self.eval(**bindings)

                points.append(point)
                values.append(val)
                attempts = 0  # Reset on success

            except (ValueError, ZeroDivisionError, ArithmeticError):
                # Function undefined at this point
                if self.check_undefined_points and num_undefined < self.max_undefined:
                    points.append(point)
                    values.append(None)
                    num_undefined += 1

                attempts += 1

        has_error = attempts >= max_attempts

        if has_error and not no_errors:
            raise ValueError(
                f"Cannot generate enough valid test points for formula: {self.to_string()}")

        # Cache results if this was automatic generation
        if num_points == self._num_test_points:
            self._test_points = points
            self._test_values = values

        return points, values, has_error

    def create_point_values(
        self,
        points: list[list[float]] | None = None,
        show_error: bool = True,
        cache_results: bool = False,
    ) -> list[MathValue | None]:
        """
        Evaluate formula at given test points.

        Args:
            points: Test points to evaluate at (or use cached/generate)
            show_error: If True, raise error on undefined points
            cache_results: If True, cache the results

        Returns:
            List of evaluated values (None for undefined)

        Reference: lib/Value/Formula.pm::createPointValues (lines 265-299)
        """
        if points is None:
            if self._test_points is None:
                points, _, _ = self.create_random_points()
            else:
                points = self._test_points

        values = []

        for point in points:
            try:
                bindings = {var: point[i]
                            for i, var in enumerate(self.variables)}
                val = self.eval(**bindings)
                values.append(val)

            except (ValueError, ZeroDivisionError, ArithmeticError) as e:
                if show_error and not self.check_undefined_points:
                    raise ValueError(
                        f"Cannot evaluate formula at point {point}: {e}")
                values.append(None)

        if cache_results:
            self._test_points = points
            self._test_values = values

        return values

    def python_function(self) -> Callable:
        """
        Convert formula to executable Python function.

        Returns a function that takes variable values and returns result.
        Caches the function for repeated use.

        Returns:
            Python function with signature func(*args) where args are variable values

        Reference: lib/Value/Formula.pm::perlFunction (lines 500+)
        """
        if self._python_func is not None:
            return self._python_func

        # Create function using SymPy lambdify
        if SYMPY_AVAILABLE and self._sympy_expr is not None:
            symbols = [sp.Symbol(var) for var in self.variables]
            func = sp.lambdify(symbols, self._sympy_expr,
                               modules=['numpy', 'math'])
            self._python_func = func
            return func
        else:
            # Fallback: create function using eval
            def eval_func(*args):
                bindings = {var: args[i]
                            for i, var in enumerate(self.variables)}
                return self.eval(**bindings).to_python()

            self._python_func = eval_func
            return eval_func

    # MathValue interface implementation

    def promote(self, other: MathValue) -> MathValue:
        """
        Formula is the highest precedence type, so no promotion needed.
        """
        return self

    def compare(
        self, other: MathValue, tolerance: float = 0.001, mode: str = ToleranceMode.RELATIVE
    ) -> bool:
        """
        Compare two formulas for equality using test point evaluation.

        Matches Perl's comparison behavior:
        1. Try symbolic comparison (SymPy) first
        2. Generate test points for left formula
        3. Evaluate both formulas at test points
        4. Compare values at each point with tolerance
        5. Track domain mismatches (where one formula undefined)

        Args:
            other: Other value to compare
            tolerance: Tolerance for numeric comparison
            mode: Tolerance mode

        Returns:
            True if formulas are equivalent at all test points

        Reference: lib/Value/Formula.pm::compare (lines 169-235)
        """
        # Special handling for assignment formulas
        # Reference: macros/parsers/parserAssignment.pl::compare (lines 385-403)
        if hasattr(self, '_is_assignment') and self._is_assignment:
            # This is an assignment formula - compare by right-hand sides
            if not isinstance(other, Formula):
                # Try to parse other as assignment if it's a string
                if isinstance(other, str):
                    if self.context and self.context.has_assignment_operator():
                        from pg.math.compute import _parse_assignment
                        other = _parse_assignment(other, self.context)
                        if other is None:
                            return False
                    else:
                        return False
                else:
                    return False
            
            # Check if other is also an assignment
            if not (hasattr(other, '_is_assignment') and other._is_assignment):
                return False
            
            # Get assignment values
            self_assign = getattr(self, '_assignment_value', None)
            other_assign = getattr(other, '_assignment_value', None)
            
            if self_assign is None or other_assign is None:
                return False
            
            # Handle function assignment parameter renaming
            # Reference: Perl lines 390-400
            if self_assign.is_function and other_assign.is_function:
                # Both are function assignments
                if len(self_assign.params) != len(other_assign.params):
                    return False
                
                # Parameter names can differ, but we compare RHS values
                # For now, just compare RHS values directly
                # TODO: Implement parameter substitution for proper comparison
                return self_assign.value.compare(other_assign.value, tolerance, mode) if hasattr(self_assign.value, 'compare') else self_assign.value == other_assign.value
            
            # For variable assignments, variable names must match
            if not self_assign.is_function and not other_assign.is_function:
                if self_assign.variable != other_assign.variable:
                    return False
                # Compare RHS values
                return self_assign.value.compare(other_assign.value, tolerance, mode) if hasattr(self_assign.value, 'compare') else self_assign.value == other_assign.value
            
            # One is function, one is variable - not equal
            return False
        
        if not isinstance(other, Formula):
            # Try to promote other to Formula
            if isinstance(other, MathValue):
                other = Formula(str(other.to_python()), [], self.context)
            else:
                return False

        # Try symbolic comparison first (fast path)
        if SYMPY_AVAILABLE and self._sympy_expr is not None and other._sympy_expr is not None:
            try:
                difference = sp.simplify(self._sympy_expr - other._sympy_expr)
                if difference == 0:
                    self.domain_mismatch = False
                    return True
            except Exception:
                pass  # Fall through to test point evaluation

        # Get or generate test points for left formula
        if self._test_points is None or self._test_values is None:
            points, left_values, _ = self.create_random_points(no_errors=True)
        else:
            points = self._test_points
            left_values = self._test_values

        # Evaluate right formula at same test points
        try:
            right_values = other.create_point_values(points, show_error=False)
        except Exception:
            # Right formula can't be evaluated
            self.domain_mismatch = True
            return False

        # Compare values at each test point
        self.domain_mismatch = False

        for left_val, right_val in zip(left_values, right_values):
            # Check for domain mismatch (one defined, other not)
            if (left_val is None) != (right_val is None):
                self.domain_mismatch = True
                continue

            # Skip if both undefined
            if left_val is None and right_val is None:
                continue

            # Compare defined values
            if not left_val.compare(right_val, tolerance, mode):
                return False

        return True

    def to_string(self) -> str:
        """Convert to human-readable string."""
        if self._sympy_expr is not None:
            return str(self._sympy_expr)
        else:
            return str(self.expression)

    def to_tex(self) -> str:
        """Convert to LaTeX representation."""
        if SYMPY_AVAILABLE and self._sympy_expr is not None:
            return sp.latex(self._sympy_expr)
        else:
            # Fallback: use pg_parser TeXVisitor
            try:
                from pg.parser.parser import Parser
                from pg.parser.visitors import TeXVisitor

                parser = Parser(self.context)
                ast = parser.parse(self.expression)
                visitor = TeXVisitor(self.context)
                return ast.accept(visitor)
            except ImportError:
                return str(self.expression)

    def to_python(self) -> Any:
        """
        Convert to Python representation.

        If formula has no variables, returns the evaluated value.
        Otherwise, returns the expression string.
        """
        if not self.variables:
            try:
                return self.eval().to_python()
            except Exception:
                pass

        return self.to_string()

    # Operator overloading

    def __add__(self, other: Any) -> Formula:
        """Addition: self + other"""
        if not isinstance(other, MathValue):
            from .numeric import Real

            other = Real(float(other))

        if SYMPY_AVAILABLE and self._sympy_expr is not None:
            if isinstance(other, Formula) and other._sympy_expr is not None:
                result = self._sympy_expr + other._sympy_expr
            else:
                result = self._sympy_expr + other.to_python()

            # Combine variable lists
            combined_vars = list(set(self.variables) | set(
                getattr(other, "variables", [])))
            return Formula(result, combined_vars, self.context)
        else:
            # Fallback: string concatenation
            return Formula(f"({self.to_string()}) + ({other.to_string()})", self.variables, self.context)

    def __radd__(self, other: Any) -> Formula:
        """Right addition: other + self"""
        return self.__add__(other)

    def __sub__(self, other: Any) -> Formula:
        """Subtraction: self - other"""
        if not isinstance(other, MathValue):
            from .numeric import Real

            other = Real(float(other))

        if SYMPY_AVAILABLE and self._sympy_expr is not None:
            if isinstance(other, Formula) and other._sympy_expr is not None:
                result = self._sympy_expr - other._sympy_expr
            else:
                result = self._sympy_expr - other.to_python()

            combined_vars = list(set(self.variables) | set(
                getattr(other, "variables", [])))
            return Formula(result, combined_vars, self.context)
        else:
            return Formula(f"({self.to_string()}) - ({other.to_string()})", self.variables, self.context)

    def __rsub__(self, other: Any) -> Formula:
        """Right subtraction: other - self"""
        if not isinstance(other, MathValue):
            from .numeric import Real

            other = Real(float(other))

        return Formula.from_math_value(other).__sub__(self)

    def __mul__(self, other: Any) -> Formula:
        """Multiplication: self * other"""
        if not isinstance(other, MathValue):
            from .numeric import Real

            other = Real(float(other))

        if SYMPY_AVAILABLE and self._sympy_expr is not None:
            if isinstance(other, Formula) and other._sympy_expr is not None:
                result = self._sympy_expr * other._sympy_expr
            else:
                result = self._sympy_expr * other.to_python()

            combined_vars = list(set(self.variables) | set(
                getattr(other, "variables", [])))
            return Formula(result, combined_vars, self.context)
        else:
            return Formula(f"({self.to_string()}) * ({other.to_string()})", self.variables, self.context)

    def __rmul__(self, other: Any) -> Formula:
        """Right multiplication: other * self"""
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Formula:
        """Division: self / other"""
        if not isinstance(other, MathValue):
            from .numeric import Real

            other = Real(float(other))

        if SYMPY_AVAILABLE and self._sympy_expr is not None:
            if isinstance(other, Formula) and other._sympy_expr is not None:
                result = self._sympy_expr / other._sympy_expr
            else:
                result = self._sympy_expr / other.to_python()

            combined_vars = list(set(self.variables) | set(
                getattr(other, "variables", [])))
            return Formula(result, combined_vars, self.context)
        else:
            return Formula(f"({self.to_string()}) / ({other.to_string()})", self.variables, self.context)

    def __rtruediv__(self, other: Any) -> Formula:
        """Right division: other / self"""
        if not isinstance(other, MathValue):
            from .numeric import Real

            other = Real(float(other))

        return Formula.from_math_value(other).__truediv__(self)

    def __pow__(self, other: Any) -> Formula:
        """Exponentiation: self ** other"""
        if not isinstance(other, MathValue):
            from .numeric import Real

            other = Real(float(other))

        if SYMPY_AVAILABLE and self._sympy_expr is not None:
            if isinstance(other, Formula) and other._sympy_expr is not None:
                result = self._sympy_expr ** other._sympy_expr
            else:
                result = self._sympy_expr ** other.to_python()

            combined_vars = list(set(self.variables) | set(
                getattr(other, "variables", [])))
            return Formula(result, combined_vars, self.context)
        else:
            return Formula(f"({self.to_string()}) ** ({other.to_string()})", self.variables, self.context)

    def __rpow__(self, other: Any) -> Formula:
        """Right exponentiation: other ** self"""
        if not isinstance(other, MathValue):
            from .numeric import Real

            other = Real(float(other))

        return Formula.from_math_value(other).__pow__(self)

    def __neg__(self) -> Formula:
        """Unary negation: -self"""
        if SYMPY_AVAILABLE and self._sympy_expr is not None:
            return Formula(-self._sympy_expr, self.variables, self.context)
        else:
            return Formula(f"-({self.to_string()})", self.variables, self.context)

    def __pos__(self) -> Formula:
        """Unary positive: +self"""
        return self

    def __abs__(self) -> Formula:
        """Absolute value: abs(self)"""
        if SYMPY_AVAILABLE and self._sympy_expr is not None:
            return Formula(sp.Abs(self._sympy_expr), self.variables, self.context)
        else:
            return Formula(f"abs({self.to_string()})", self.variables, self.context)

    @classmethod
    def from_math_value(cls, value: MathValue) -> Formula:
        """Convert a MathValue to a Formula."""
        if isinstance(value, Formula):
            return value

        if SYMPY_AVAILABLE:
            python_val = value.to_python()
            if isinstance(python_val, (int, float)):
                return cls(sp.sympify(python_val), [], None)
            else:
                return cls(str(python_val), [], None)
        else:
            return cls(str(value.to_python()), [], None)

    # Answer checking (NEW FOR PARITY)

    @property
    def cmp(self):
        """
        Create an answer evaluator for this formula.

        This is the built-in answer checker that every Formula has in Perl.
        Returns a CMPWrapper that can be:
        - Called as cmp() to get answer checker
        - Chained as cmp.withPostFilter(...) for Perl compatibility

        Returns:
            CMPWrapper that works as both callable and has withPostFilter

        Example:
            >>> f = Formula("x^2", variables=["x"])
            >>> checker = f.cmp()
            >>> result = checker.check("x*x")
            >>> result['correct']  # True
            >>> # Or chain: f.cmp.withPostFilter(AnswerHints(...))

        Reference: lib/Value/Formula.pm::cmp (lines 430-470)
        """
        return CMPWrapper(self)

    def adapt_parameters(self, student_formula, *param_names):
        """Adaptive parameter finding (advanced)."""
        if not SYMPY_AVAILABLE:
            return None

        n_params = len(param_names)
        if n_params == 0:
            return {}

        try:
            import numpy as np

            regular_vars = [v for v in self.variables if v not in param_names]
            if not regular_vars:
                return None

            test_pts, _, has_error = self.create_random_points(
                num_points=n_params, include=regular_vars
            )
            if has_error or not test_pts:
                return None

            student_func = student_formula.python_function()
            correct_func = self.python_function()

            A_matrix = []
            b_vector = []

            for pt in test_pts:
                try:
                    student_val = student_func(*pt)
                    row = []
                    for param in param_names:
                        param_pt = list(
                            pt) + [1 if p == param else 0 for p in param_names]
                        val_with = correct_func(*param_pt)
                        val_without = correct_func(*list(pt), *[0]*n_params)
                        row.append(float(val_with - val_without))
                    baseline = correct_func(*list(pt), *[0]*n_params)
                    b_vector.append(float(student_val - baseline))
                    A_matrix.append(row)
                except:
                    return None

            params_solution = np.linalg.lstsq(
                np.array(A_matrix), np.array(b_vector), rcond=None)[0]
            return {param: float(val) for param, val in zip(param_names, params_solution)}
        except:
            return None

    def toUnits(self, unit_string):
        """
        Convert this formula with units to a specified unit.

        Args:
            unit_string: Target unit as string (e.g., 'cup', 'mi/h')

        Returns:
            New Formula with converted units (stub implementation)
        """
        # Stub: In full implementation, this would use unit conversion
        # For now, return self as-is
        return self

    def __getitem__(self, key):
        """
        Support subscript access for Perl hash-like attribute access.

        Allows: formula['test_points'] or formula['limits']

        This enables Perl idiom: $f->{test_points} = [[1], [2]]
        which preprocessor converts to: f['test_points'] = [[1], [2]]
        """
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """
        Support subscript assignment for Perl hash-like attribute setting.

        Allows: formula['test_points'] = [[1], [2], [3]]

        This enables Perl idiom: $f->{test_points} = [[1], [2]]
        which preprocessor converts to: f['test_points'] = [[1], [2]]
        """
        setattr(self, key, value)
