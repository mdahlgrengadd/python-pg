r"""
Assignment Parser for WeBWorK.

Provides parsing for assignment expressions (e.g., x = 5, f(x) = x^2),
useful in equation-solving and function definition problems.

Reference: macros/parsers/parserAssignment.pl

Example:
    Context('Numeric')->variables->add(y => 'Real');
    parser::Assignment->Allow;

    $eqn = Formula('y = 3x + 1');

Or with function declarations:
    parser::Assignment->Function('f', 'g');
    $f = Formula('f(x) = x^2');
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


class AssignmentValue:
    """Represents an assignment value with variable and expression."""

    def __init__(self, variable: str, value: Any, is_function: bool = False, params: Optional[List[str]] = None):
        """
        Create an assignment value.

        Args:
            variable: Variable or function name being assigned
            value: The right-hand side value
            is_function: Whether this is a function assignment
            params: List of parameter names for function assignments
        """
        self.variable = variable
        self.value = value
        self.is_function = is_function
        self.params = params or []

    def string(self) -> str:
        """Return string representation."""
        if self.is_function:
            params_str = ",".join(self.params)
            return f"{self.variable}({params_str}) = {self.value}"
        return f"{self.variable} = {self.value}"

    def tex(self) -> str:
        """Return LaTeX representation."""
        if self.is_function:
            params_str = ",".join(self.params)
            return f"{self.variable}({params_str}) = {self.value}"
        return f"{self.variable} = {self.value}"

    def type_ref(self) -> str:
        """Return the type reference (based on RHS value type)."""
        return "Assignment"

    def cmp_class(self) -> str:
        """Return description for answer checker."""
        type_name = "Function" if self.is_function else "Variable"
        return f"a {type_name} equal to {self.value}"

    def type_match(self, other: "AssignmentValue", ans: Dict[str, Any] = None) -> bool:
        """
        Check if two assignments have compatible types.
        Type matching is based on RHS values, not LHS variable names.
        """
        if not isinstance(other, AssignmentValue):
            return False
        # Compare right-hand side types
        return type(self.value) == type(other.value)

    def compare(self, other: "AssignmentValue") -> bool:
        """
        Compare two assignments.
        For function assignments, allows parameter renaming.
        """
        # Both must be same kind (function vs variable)
        if self.is_function != other.is_function:
            return False

        # For function assignments, check parameter count matches
        if self.is_function:
            if len(self.params) != len(other.params):
                return False
            # Would need to rename parameters and compare RHS
            # For now, just compare values
            return self.value == other.value

        # For variable assignments, variable names must match
        if self.variable != other.variable:
            return False

        # Compare RHS values
        return self.value == other.value


class AssignmentBOP:
    """
    Binary operator for assignment expressions.

    Represents the '=' operator used in assignment expressions.
    Validates that LHS is a variable or function and RHS is any value.
    """

    def __init__(self, lop: Any = None, rop: Any = None):
        """Initialize assignment BOP."""
        self.lop = lop  # Left operand (variable or function)
        self.rop = rop  # Right operand (any value)
        self.type = "Assignment"
        self.is_function = False

    def _check(self) -> None:
        """
        Validate assignment structure.

        Checks:
        1. No nested assignments on either side
        2. LHS must be variable or function declaration
        3. For function assignments: params must be variables, all different
        4. For variable assignments: RHS can't contain assigned variable
        """
        # Check for nested assignments
        if hasattr(self.lop, "type") and self.lop.type == "Assignment":
            raise ValueError("Only one assignment is allowed in an equation")
        if hasattr(self.rop, "type") and self.rop.type == "Assignment":
            raise ValueError("Only one assignment is allowed in an equation")

        # Determine if function or variable assignment
        self.is_function = hasattr(self.lop, "is_function") and self.lop.is_function

        if self.is_function:
            # For function assignments: validate parameters
            if not hasattr(self.lop, "params"):
                raise ValueError("The left side of an assignment must be a variable or function")

            # Check all parameters are variables
            if not all(isinstance(p, str) for p in self.lop.params):
                raise ValueError(f"The arguments of '{self.lop.name}' must be variables")

            # Check all parameters are different
            seen = set()
            for param in self.lop.params:
                if param in seen:
                    raise ValueError(f"The arguments of '{self.lop.name}' must all be different")
                seen.add(param)
        else:
            # For variable assignments: check LHS is variable
            if not hasattr(self.lop, "name"):
                raise ValueError("The left side of an assignment must be a variable or function")

            # Check RHS doesn't use the assigned variable
            if hasattr(self.rop, "getVariables"):
                rop_vars = self.rop.getVariables()
                if self.lop.name in rop_vars:
                    raise ValueError(
                        f"The right side of an assignment must not include the variable being defined"
                    )

    def eval(self) -> AssignmentValue:
        """Evaluate to create an AssignmentValue."""
        if self.is_function:
            # Function assignment: f(x) = expr
            return AssignmentValue(
                self.lop.name, self.rop, is_function=True, params=self.lop.params
            )
        else:
            # Variable assignment: x = expr
            return AssignmentValue(self.lop.name, self.rop, is_function=False)

    def get_variables(self) -> Set[str]:
        """Return variables used (excluding LHS variable for simple assignments)."""
        if self.is_function:
            # Include function parameters
            if hasattr(self.rop, "getVariables"):
                return self.rop.getVariables()
            return set(self.lop.params)
        else:
            # Exclude assigned variable from RHS variables
            if hasattr(self.rop, "getVariables"):
                vars_set = self.rop.getVariables()
                vars_set.discard(self.lop.name)
                return vars_set
            return set()

    def string(self) -> str:
        """Return string representation."""
        lop_str = self.lop.string() if hasattr(self.lop, "string") else str(self.lop)
        rop_str = self.rop.string() if hasattr(self.rop, "string") else str(self.rop)
        return f"{lop_str} = {rop_str}"

    def tex(self) -> str:
        """Return LaTeX representation."""
        lop_tex = self.lop.tex() if hasattr(self.lop, "tex") else str(self.lop)
        rop_tex = self.rop.tex() if hasattr(self.rop, "tex") else str(self.rop)
        return f"{lop_tex} = {rop_tex}"


class AssignmentFunction:
    """
    Dummy function class for function declarations.

    Used for assignments like f(x) = x^2.
    Cannot be evaluated, only used in assignment LHS.
    """

    def __init__(self, name: str, params: List[str]):
        """
        Initialize assignment function.

        Args:
            name: Function name (e.g., 'f', 'g')
            params: List of parameter names (e.g., ['x'], ['x', 'y'])
        """
        self.name = name
        self.params = params
        self.is_function = True
        self.is_dummy = True

    def getVariables(self) -> Set[str]:
        """Return parameter names as variables."""
        return set(self.params)

    def string(self) -> str:
        """Return string representation."""
        params_str = ",".join(self.params)
        return f"{self.name}({params_str})"

    def tex(self) -> str:
        """Return LaTeX representation."""
        params_str = ",".join(self.params)
        return f"{self.name}({params_str})"

    def eval(self):
        """Cannot be evaluated."""
        raise ValueError(f"Dummy function '{self.name}' cannot be evaluated")

    def call(self, *args):
        """Cannot be called."""
        raise ValueError(f"Dummy function '{self.name}' cannot be called")


class AssignmentParser:
    """Main class for assignment parsing and context integration."""

    # Track registered function names globally
    _registered_functions: Dict[str, List[str]] = {}

    @classmethod
    def Allow(cls, allow: bool = True, context: Optional[Any] = None) -> None:
        """
        Enable or disable assignment operator in a context.

        Reference: macros/parsers/parserAssignment.pl::Allow (lines 210-233)

        Args:
            allow: True to enable, False to disable
            context: Context to modify (if None, uses current context)
        """
        if context is None:
            from pg.math.context import get_current_context
            context = get_current_context()
        
        if allow:
            # Get comma precedence (or default to 1)
            comma_op = context.operators.get(',')
            comma_prec = comma_op.get('precedence', 1) if comma_op else 1
            
            # Register '=' operator with precedence just above comma
            context.operators.add(
                '=',
                class_name='parser::Assignment',
                precedence=comma_prec + 0.25,  # Just above comma
                associativity='left',
                type='bin',  # Binary operator
                string=' = ',
            )
            
            # Store assignment-enabled flag
            context.flags.set(assignmentEnabled=True)
        else:
            # Remove '=' operator from context
            context.operators.remove('=')
            context.flags.set(assignmentEnabled=False)

    @classmethod
    def Function(cls, *function_names: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Register function names that can appear on assignment LHS.

        Args:
            *function_names: Names of functions to register (e.g., 'f', 'g')
            context: Context to modify (if None, uses current context)
        """
        if not function_names:
            raise ValueError("You must provide a function name")

        for func_name in function_names:
            # Validate function name
            if not cls._is_valid_function_name(func_name):
                raise ValueError(f"Function name '{func_name}' is illegal")

            # Would register in context as AssignmentFunction class
            if context is None:
                cls._registered_functions.setdefault("__default__", []).append(func_name)
            else:
                # Would add to context's function registry
                pass

    @staticmethod
    def _is_valid_function_name(name: str) -> bool:
        """Check if function name is valid (alphanumeric, starts with letter)."""
        import re
        return bool(re.match(r"^[a-z][a-z0-9]*$", name, re.IGNORECASE))


# Module-level convenience functions
def Assignment(*args: Any) -> AssignmentValue:
    """Create an AssignmentValue from arguments."""
    if len(args) == 2:
        return AssignmentValue(args[0], args[1])
    elif len(args) == 3:
        return AssignmentValue(args[0], args[1], is_function=args[2])
    else:
        raise ValueError("Assignment requires 2 or 3 arguments")


__all__ = [
    "AssignmentValue",
    "AssignmentBOP",
    "AssignmentFunction",
    "AssignmentParser",
    "Assignment",
]

