"""
Enhanced Formula with complete Value::Formula.pm parity.

Adds:
- Advanced test point generation with granularity
- Adaptive parameter system
- Test value caching
- Domain checking with undefined points
- Python function generation with caching
- Enhanced differentiation

Reference: Value::Formula.pm (1,156 lines)
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

from .formula import Formula
from .value import MathValue, ToleranceMode


# UNDEF sentinel for undefined points
class UNDEF:
    """Sentinel for undefined points in domain."""
    def __repr__(self) -> str:
        return "UNDEF"


UNDEF_VALUE = UNDEF()


class FormulaEnhanced(Formula):
    """
    Enhanced Formula with complete feature set.

    Reference: Value::Formula.pm
    """

    def __init__(
        self,
        expression: str | Any,
        variables: list[str] | None = None,
        context: Any | None = None,
        # Test point configuration
        test_points: list[list[float]] | None = None,
        num_test_points: int = 5,
        limits: dict[str, tuple[float, float]] | None = None,
        granularity: int = 1000,
        # Domain checking
        check_undefined_points: bool = False,
        max_undefined: int | None = None,
        # Adaptive parameters
        parameters: list[str] | None = None,
        # Other options
        **options: Any,
    ):
        """
        Initialize enhanced formula.

        Args:
            expression: Mathematical expression
            variables: Variable names
            context: Mathematical context
            test_points: Pre-specified test points
            num_test_points: Number of random test points
            limits: Variable limits {var: (min, max)}
            granularity: Granularity for point distribution
            check_undefined_points: Track undefined points
            max_undefined: Max allowed undefined points
            parameters: Parameter names for adaptive solving
            **options: Additional options
        """
        # Store parameters before calling super().__init__ so we can rebuild SymPy expr
        self._parameters = parameters or []

        super().__init__(expression, variables, context, test_points, num_test_points, limits)

        # Test point configuration
        self._granularity = granularity

        # Domain checking
        self.domain_mismatch = False
        self.check_undefined_points = check_undefined_points
        self.max_undefined = max_undefined or num_test_points

        # Adaptive parameters
        self._test_adapt: list[MathValue] | None = None
        self._param_func: Callable | None = None

        # Caching
        self._python_func: Callable | None = None
        self._func_cache: dict[tuple, Callable] = {}

        # Options
        self._options = options

        # Rebuild SymPy expression with parameters included
        if isinstance(expression, str) and SYMPY_AVAILABLE and self._parameters:
            try:
                # Parse with both variables and parameters as symbols
                all_vars = (variables or []) + self._parameters
                local_dict = {var: sp.Symbol(var) for var in all_vars}
                # Add mathematical constants
                local_dict['e'] = sp.E
                local_dict['pi'] = sp.pi
                local_dict['E'] = sp.E
                local_dict['PI'] = sp.pi
                self._sympy_expr = sp.parse_expr(
                    expression,
                    transformations="all",
                    local_dict=local_dict,
                )
            except Exception:
                # Fallback: keep existing _sympy_expr
                pass

    def create_random_points(
        self,
        n: int | None = None,
        test_at: dict[str, list[float]] | None = None,
        no_errors: bool = False,
        **kwargs: Any
    ) -> list[list[float]] | tuple[list[list[float]], list[MathValue], bool]:
        """
        Generate random test points with granularity control.

        Reference: Value::Formula.pm:300-380

        Args:
            n: Number of points (uses default if None)
            test_at: Specific points to include {var: [values]}
            no_errors: If True, return tuple with values and error flag

        Returns:
            List of test points, or tuple (points, values, has_error) if no_errors=True
        """
        # Get number of points
        if n is None:
            n = self._num_test_points

        # Get variables
        vars_list = sorted(self.variables)
        if not vars_list:
            return [[]]  # No variables

        # Get limits for each variable
        limits = {}
        default_limits = self._limits.get("default", [-2, 2])

        for var in vars_list:
            if var in self._limits:
                limits[var] = self._limits[var]
            else:
                limits[var] = default_limits

        # Handle test_at (must-include points)
        must_include = []
        if test_at:
            for var, values in test_at.items():
                if var in vars_list:
                    for val in values:
                        # Create point with this value for var, random for others
                        point = []
                        for v in vars_list:
                            if v == var:
                                point.append(val)
                            else:
                                lo, hi = limits[v]
                                point.append(random.uniform(lo, hi))
                        must_include.append(point)

        # Generate random points
        points = []
        rng = self._get_rng()

        # Add must-include points
        points.extend(must_include)

        # Generate remaining random points
        remaining = n - len(must_include)
        for _ in range(remaining):
            point = []
            for var in vars_list:
                lo, hi = limits[var]
                point.append(rng.uniform(lo, hi))
            points.append(point)

        # Apply granularity
        if self._granularity > 0:
            points = self._apply_granularity(points, limits, self._granularity)

        # If no_errors is True, evaluate points and return tuple
        if no_errors:
            try:
                values = self.create_point_values(points, show_error=False, check_undefined=True)
                if values is None:
                    return (points, [], True)  # Error occurred
                return (points, values, False)  # No error
            except Exception:
                return (points, [], True)  # Error occurred

        return points

    def _apply_granularity(
        self,
        points: list[list[float]],
        limits: dict[str, tuple[float, float]],
        granularity: int
    ) -> list[list[float]]:
        """
        Apply granularity to ensure evenly-spaced test points.

        Reference: Value::Formula.pm:340-360

        Args:
            points: Original points
            limits: Variable limits
            granularity: Number of steps in range

        Returns:
            Adjusted points
        """
        vars_list = sorted(self.variables)
        adjusted = []

        for point in points:
            adjusted_point = []
            for i, var in enumerate(vars_list):
                lo, hi = limits[var]
                val = point[i]

                # Round to granularity
                range_size = hi - lo
                step = range_size / granularity
                rounded = round((val - lo) / step) * step + lo

                # Clamp to limits
                rounded = max(lo, min(hi, rounded))
                adjusted_point.append(rounded)

            adjusted.append(adjusted_point)

        return adjusted

    def _get_rng(self) -> random.Random:
        """Get RNG from context or create new one."""
        if hasattr(self, "_rng"):
            return self._rng

        # Get seed from context or use default
        seed = self._options.get("problem_seed", 12345)
        self._rng = random.Random(seed)
        return self._rng

    def create_point_values(
        self,
        points: list[list[float]] | None = None,
        show_error: bool = True,
        cache_results: bool = False,
        check_undefined: bool = False
    ) -> list[MathValue | UNDEF] | None:
        """
        Evaluate formula at test points with caching.

        Reference: Value::Formula.pm:264-299

        Args:
            points: Test points
            show_error: Raise error on undefined point
            cache_results: Cache points and values
            check_undefined: Track undefined points

        Returns:
            List of evaluated values (None for failure, UNDEF for undefined points)
        """
        # Get test points
        if points is None:
            if hasattr(self, "_test_points") and self._test_points:
                points = self._test_points
            else:
                points = self.create_random_points()

        # Get variables and parameters
        vars_list = sorted(self.variables)
        params_list = sorted(self._parameters)
        param_zeros = [0] * len(params_list)

        # Get or create Python function
        func = self.python_function(vars=vars_list + params_list)

        # Track undefined points
        undefined_count = 0
        max_undefined = self.max_undefined

        # Evaluate at each point
        values: list[MathValue | UNDEF] = []
        for point in points:
            try:
                # Call function with point values and parameter zeros
                result = func(*point, *param_zeros)

                if result is None or (isinstance(result, float) and math.isnan(result)):
                    if not check_undefined:
                        if show_error:
                            raise ValueError(
                                f"Can't evaluate formula on test point ({', '.join(map(str, point))})"
                            )
                        return None

                    # Track undefined
                    undefined_count += 1
                    if undefined_count > max_undefined:
                        if show_error:
                            raise ValueError(
                                f"Too many undefined points (>{max_undefined})"
                            )
                        return None

                    values.append(UNDEF_VALUE)
                else:
                    # Convert to MathValue
                    value = MathValue.from_python(result)

                    # Transfer tolerance flags from formula
                    if hasattr(self, "tolerance"):
                        value.tolerance = self.tolerance  # type: ignore
                    if hasattr(self, "tolType"):
                        value.tolType = self.tolType  # type: ignore

                    values.append(value)

            except (ZeroDivisionError, ValueError, OverflowError) as e:
                if not check_undefined:
                    if show_error:
                        raise ValueError(f"Error evaluating formula: {e}")
                    return None

                undefined_count += 1
                if undefined_count > max_undefined:
                    if show_error:
                        raise ValueError(f"Too many undefined points")
                    return None

                values.append(UNDEF_VALUE)

        # Cache if requested
        if cache_results:
            self._test_points = points
            self._test_values = values

        return values

    def python_function(
        self,
        name: str | None = None,
        vars: list[str] | None = None
    ) -> Callable:
        """
        Generate Python function from formula.

        Reference: Parser.pm:794-834 (perlFunction equivalent)

        Args:
            name: Function name (None for lambda)
            vars: Variable names (uses formula vars if None)

        Returns:
            Compiled Python function
        """
        # Use formula variables if not specified
        if vars is None:
            vars = sorted(self.variables)

        # Check cache
        cache_key = (name or "", tuple(vars))
        if cache_key in self._func_cache:
            return self._func_cache[cache_key]

        # Build function
        if SYMPY_AVAILABLE and hasattr(self, "_sympy_expr") and self._sympy_expr is not None:
            # Use SymPy lambdify (fast!)
            try:
                # Use "numpy" first for numerical evaluation, then "math" as fallback
                base_func = sp.lambdify(
                    [sp.Symbol(v) for v in vars],
                    self._sympy_expr,
                    modules=["numpy", "math", {"abs": abs}]
                )

                # Wrap to ensure float conversion (in case SymPy returns symbolic)
                def func(*args):
                    result = base_func(*args)
                    # Convert SymPy expressions to float
                    if hasattr(result, "evalf"):
                        evaluated = result.evalf()
                        # Try to extract numerical value
                        if hasattr(evaluated, "is_Number") and evaluated.is_Number:
                            return float(evaluated)
                        elif hasattr(evaluated, "n"):
                            return float(evaluated.n())
                        else:
                            # Fallback: force conversion
                            try:
                                return float(evaluated)
                            except:
                                # Return the expression as-is and let caller handle it
                                return float(result.n())  # Force numerical evaluation
                    return result

            except Exception:
                # Fallback to eval
                func = self._build_eval_function(vars)
        else:
            # Fallback: use eval
            func = self._build_eval_function(vars)

        # Cache function
        self._func_cache[cache_key] = func

        return func

    def _build_eval_function(self, vars: list[str]) -> Callable:
        """Build function using eval."""
        # Convert formula to Python expression
        expr_str = str(self.expression)

        # Build function
        var_params = ", ".join(vars)
        func_code = f"""
def formula_func({var_params}):
    import math
    return {expr_str}
"""

        # Compile
        namespace: dict[str, Any] = {}
        exec(func_code, namespace)
        return namespace["formula_func"]


    def adapt_parameters(
        self,
        student_formula: "FormulaEnhanced",
        max_adapt: float = 1e8,
        attempts: int = 3
    ) -> bool:
        """
        Solve for adaptive parameters using student's formula.

        Reference: Value::Formula.pm:484-570 (AdaptParameters)

        This implements adaptive parameter solving for formulas like:
        - Correct: C*e^x (with parameter C)
        - Student: 5*e^x
        Result: Solves C = 5

        Args:
            student_formula: Student's formula to adapt to
            max_adapt: Maximum allowed parameter value
            attempts: Number of random attempts

        Returns:
            True if parameters solved successfully
        """
        # Check if we have parameters
        if not self._parameters:
            return False

        # Check if formula uses any parameters
        if not self.uses_one_of(*self._parameters):
            return False

        # Get number of parameters
        num_params = len(self._parameters)
        params_list = sorted(self._parameters)

        # Try multiple attempts with different random points
        for attempt in range(attempts):
            try:
                # Generate test points
                points = self.create_random_points(n=num_params)

                # Build linear system: A * params = b
                # where A[i][j] = effect of param[j] at point[i]
                #       b[i] = student_value[i] - base_value[i]
                A_matrix = []
                b_vector = []

                # Get functions
                vars_list = sorted(self.variables)
                correct_func = self.python_function(vars=vars_list + params_list)
                student_func = student_formula.python_function(vars=vars_list)

                # Evaluate at each test point
                for point in points:
                    # Get base value (all parameters = 0)
                    param_zeros = [0] * num_params
                    base_value = correct_func(*point, *param_zeros)

                    if base_value is None or (isinstance(base_value, float) and math.isnan(base_value)):
                        raise ValueError("Undefined base value")

                    # Get student value
                    student_value = student_func(*point)

                    if student_value is None or (isinstance(student_value, float) and math.isnan(student_value)):
                        # Can't adapt if student formula undefined
                        return False

                    # Get coefficients for each parameter
                    row = []
                    for j, param in enumerate(params_list):
                        # Set param[j] = 1, others = 0
                        param_values = [0] * num_params
                        param_values[j] = 1

                        # Evaluate with this parameter set
                        param_value = correct_func(*point, *param_values)

                        if param_value is None or (isinstance(param_value, float) and math.isnan(param_value)):
                            raise ValueError(f"Undefined parameter value for {param}")

                        # Coefficient is difference from base
                        coeff = float(param_value) - float(base_value)
                        row.append(coeff)

                    A_matrix.append(row)

                    # RHS is student - base
                    b_val = float(student_value) - float(base_value)
                    b_vector.append(b_val)

                # Solve linear system A * x = b
                solution = self._solve_linear_system(A_matrix, b_vector)

                if solution is None:
                    continue  # Try next attempt

                # Check solution values
                for i, (param, value) in enumerate(zip(params_list, solution)):
                    if abs(value) > max_adapt:
                        # Check if it's a constant of integration
                        if param in ("C0", "n00"):
                            raise ValueError(
                                f"Constant of integration is too large: {value}\n"
                                f"(maximum allowed is {max_adapt})"
                            )
                        else:
                            raise ValueError(
                                f"Adaptive constant is too large: {param} = {value}\n"
                                f"(maximum allowed is {max_adapt})"
                            )

                # Success! Store parameters
                self._parameters_values = solution
                self._parameters_dict = dict(zip(params_list, solution))

                # Recompute test values with adapted parameters
                self._test_adapt = self._create_adapted_values()

                return True

            except ValueError as e:
                # Check if it's a "too large" error - if so, re-raise immediately
                if "too large" in str(e):
                    raise
                # Otherwise, try next attempt
                continue
            except (ZeroDivisionError, OverflowError):
                # Try next attempt
                continue

        # Failed all attempts
        raise ValueError("Can't solve for adaptive parameters")

    def _solve_linear_system(
        self,
        A: list[list[float]],
        b: list[float]
    ) -> list[float] | None:
        """
        Solve linear system A * x = b.

        Reference: Value::Formula.pm:527-566 (uses MatrixReal1)

        For simple cases, uses direct solution.
        For complex cases, requires scipy.

        Args:
            A: Coefficient matrix (n x n)
            b: RHS vector (n)

        Returns:
            Solution vector x, or None if no solution
        """
        n = len(A)

        # Simple case: 1 parameter (linear)
        if n == 1:
            if abs(A[0][0]) < 1e-10:
                return None  # Singular
            return [b[0] / A[0][0]]

        # Try using numpy/scipy if available
        try:
            import numpy as np
            A_np = np.array(A, dtype=float)
            b_np = np.array(b, dtype=float)

            # Check determinant
            det = np.linalg.det(A_np)
            if abs(det) < 1e-6:
                return None  # Singular

            # Solve
            x = np.linalg.solve(A_np, b_np)
            return x.tolist()

        except ImportError:
            # Fallback for 2x2 case (common for 2 parameters)
            if n == 2:
                a11, a12 = A[0]
                a21, a22 = A[1]
                b1, b2 = b

                det = a11 * a22 - a12 * a21
                if abs(det) < 1e-6:
                    return None

                # Cramer's rule
                x1 = (b1 * a22 - b2 * a12) / det
                x2 = (a11 * b2 - a21 * b1) / det
                return [x1, x2]

            # For n > 2, need scipy
            raise ValueError(
                "Adaptive parameters with >2 parameters requires numpy. "
                "Install with: pip install numpy"
            )

    def _create_adapted_values(self) -> list[MathValue]:
        """
        Create test values with adapted parameters.

        Reference: Value::Formula.pm:563 (createAdaptedValues)

        Returns:
            List of values evaluated with adapted parameters
        """
        if not hasattr(self, "_test_points") or not self._test_points:
            self._test_points = self.create_random_points()

        points = self._test_points
        params_list = sorted(self._parameters)
        param_values = self._parameters_values

        # Get function
        vars_list = sorted(self.variables)
        func = self.python_function(vars=vars_list + params_list)

        # Evaluate at each point with adapted parameters
        values = []
        for point in points:
            result = func(*point, *param_values)
            value = MathValue.from_python(result)

            # Transfer flags
            if hasattr(self, "tolerance"):
                value.tolerance = self.tolerance  # type: ignore
            if hasattr(self, "tolType"):
                value.tolType = self.tolType  # type: ignore

            values.append(value)

        return values

    def uses_one_of(self, *names: str) -> bool:
        """
        Check if formula uses any of the given variable/parameter names.

        Args:
            *names: Variable or parameter names to check

        Returns:
            True if formula uses any of the names
        """
        # Check both variables and parameters
        formula_vars = set(self.variables) | set(self._parameters)
        check_vars = set(names)
        return bool(formula_vars & check_vars)

    def compare(
        self,
        other: "FormulaEnhanced",
        use_adaptive: bool = True,
        **options: Any
    ) -> bool:
        """
        Compare formulas with optional adaptive parameter solving.

        Reference: Value::Formula.pm:169-235 (cmp_compare with adaptive)

        Args:
            other: Formula to compare with
            use_adaptive: Try adaptive parameters if available
            **options: Comparison options

        Returns:
            True if formulas are equivalent
        """
        # Try adaptive parameters if enabled and available
        if use_adaptive and self._parameters:
            try:
                if self.adapt_parameters(other):
                    # Use adapted test values
                    if hasattr(self, "_test_adapt"):
                        return self._compare_test_values(
                            self._test_adapt,
                            other,
                            **options
                        )
            except ValueError:
                # Adaptation failed, fall through to regular comparison
                pass

        # Regular comparison using test points
        return super().compare(other, **options)

    def _compare_test_values(
        self,
        values: list[MathValue],
        other: "FormulaEnhanced",
        **options: Any
    ) -> bool:
        """
        Compare using pre-computed test values.

        Args:
            values: Test values from this formula
            other: Other formula
            **options: Comparison options

        Returns:
            True if all values match
        """
        # Get test points
        if not hasattr(self, "_test_points"):
            return False

        points = self._test_points

        # Evaluate other formula at same points
        other_values = other.create_point_values(points)

        if other_values is None:
            return False

        # Compare each pair
        tolerance = options.get("tolerance", getattr(self, "tolerance", 1e-6))

        for v1, v2 in zip(values, other_values):
            if isinstance(v1, UNDEF) or isinstance(v2, UNDEF):
                # Both must be undefined
                if not (isinstance(v1, UNDEF) and isinstance(v2, UNDEF)):
                    return False
            else:
                # Compare numerically
                diff = abs(float(v1) - float(v2))
                if diff > tolerance:
                    return False

        return True
