"""
Function Parser for WeBWorK.

This module provides utilities for defining custom functions in parsing contexts,
allowing problems to use custom mathematical functions in student answers.

Based on parserFunction.pl from the Perl WeBWorK distribution.

Example:
    >>> parserFunction("f(x)", "sqrt(x+1)-2")
    >>> parserFunction("g(x,y)", "x*y")
    >>> parserFunction("u(t)", "step(t)")  # Define u as step function

The first argument can be:
- Just the function name: "f" -> uses variables from formula
- Function with arguments: "f(x)" or "f(x,y)" -> explicit argument names

The second argument is the formula (string or Formula object).
"""

from typing import Any, Callable, Dict, List, Optional
import re


def parserFunction(name_spec: Any = None, formula_spec: Any = None, context: Optional[Any] = None,
                   **kwargs: Any) -> None:
    """
    Define a custom function in the parsing context.

    Adds a named function to the mathematical parser context, allowing problems
    to use custom mathematical functions in student answers and comparisons.

    Supports multiple calling conventions:
    - parserFunction("f", "formula")              # f(x) auto-detected from formula
    - parserFunction("f(x)", "formula")          # f with explicit argument x
    - parserFunction("f(x,y)", "formula")        # f with explicit arguments x,y
    - parserFunction({'f(x)': 'formula'})        # Perl hash-style (from =>)

    Args:
        name_spec: Function name or name with argument list (e.g., "f", "f(x)", "f(x,y)")
                   Can also be a dict like {'f(x)': 'formula'} (from Perl 'f(x)' => 'formula')
        formula_spec: The function formula as a string or Formula object
        context: Parser context to add function to (default: current context)
        **kwargs: Additional options to pass to context.functions.add()
               Common options: TeX, limits, test_at, etc.

    Returns:
        None (modifies context in place)

    Example:
        >>> # Simple function with auto-detected variables
        >>> parserFunction("f", "x^2 + 1")

        >>> # Function with explicit arguments
        >>> parserFunction("f(x)", "x^2 + 1")
        >>> parserFunction("g(x,y)", "sqrt(x*y)")

        >>> # Using Heaviside function
        >>> parserFunction("u(t)", "step(t)")

        >>> # Perl hash-style (from preprocessor)
        >>> parserFunction({'u(t)': 'step(t)'})

    Reference: macros/parsers/parserFunction.pl
    """
    # Import here to avoid circular dependencies
    from pg.mathobjects import Context, Formula

    # Handle dict argument (from Perl 'name' => 'formula' syntax)
    if isinstance(name_spec, dict):
        if len(name_spec) != 1:
            raise ValueError(
                "parserFunction() dict argument must have exactly one key-value pair")
        name_spec, formula_spec = next(iter(name_spec.items()))

    # Handle bareword => formula pattern: parserFunction(f => 'formula')
    # The preprocessor converts this to parserFunction(f='formula') as a keyword arg
    if name_spec is None and formula_spec is None and len(kwargs) == 1:
        # Extract the single keyword argument as name => formula
        name_spec, formula_spec = next(iter(kwargs.items()))
        kwargs = {}  # Clear kwargs since we've consumed it

    # Validate that we have both name and formula
    if name_spec is None or formula_spec is None:
        raise TypeError(
            "parserFunction() requires both name and formula arguments")

    # Ensure name_spec is a string
    if not isinstance(name_spec, str):
        raise TypeError(
            f"Function name must be a string, got {type(name_spec).__name__}")

    # Get the current context if not provided
    if context is None:
        try:
            from pg.math.context import get_current_context
            context = get_current_context()
        except (ImportError, RuntimeError):
            # Fallback to default context if current context not available
            context = Context()

    # Parse the function name and extract argument names
    func_name: str = name_spec
    arg_names: List[str] = []

    # Check if name includes argument list: "f(x)" or "f(x,y)"
    # Allow letters, numbers, and underscores in function names
    name_match = re.match(
        r'^([a-z_][a-z0-9_]*)\s*\(\s*(.*?)\s*\)$', name_spec, re.IGNORECASE)

    if name_match:
        func_name = name_match.group(1)
        args_str = name_match.group(2)

        if args_str.strip():  # If there are arguments specified
            arg_names = [arg.strip() for arg in args_str.split(',')]

            # Validate argument names
            for arg_name in arg_names:
                if not re.match(r'^[a-z_][a-z0-9_]*$', arg_name, re.IGNORECASE):
                    raise ValueError(f"Illegal variable name '{arg_name}'")

                # Add variable to context if not already there
                try:
                    if arg_name not in context.variables:  # type: ignore
                        context.variables.add(arg_name, 'Real')
                except (AttributeError, TypeError):
                    # Variables might not support add method, skip
                    pass
    else:
        # Validate function name (just name, no parentheses)
        if not re.match(r'^[a-z_][a-z0-9_]*$', name_spec, re.IGNORECASE):
            raise ValueError(f"Illegal function name '{name_spec}'")

    # Convert formula_spec to Formula object if it's a string
    if isinstance(formula_spec, str):
        formula = Formula(formula_spec, context=context)
    else:
        formula = formula_spec

    # If no argument names were provided, extract them from the formula
    if not arg_names:
        # Get variables from formula and sort them alphabetically
        formula_vars: List[str] = []
        if hasattr(formula, 'get_variables'):
            formula_vars = formula.get_variables()
        elif hasattr(formula, 'variables'):
            vars_obj = formula.variables
            if isinstance(vars_obj, dict):
                formula_vars = list(vars_obj.keys())
            elif isinstance(vars_obj, (list, tuple)):
                formula_vars = list(vars_obj)
            else:
                formula_vars = []
        arg_names = sorted(formula_vars) if formula_vars else []

    # Get the argument types from the context
    arg_types: List[Any] = []
    for arg_name in arg_names:
        try:
            if arg_name in context.variables:  # type: ignore
                arg_type = context.variables[arg_name]
            else:
                arg_type = 'Real'  # Default type
        except (AttributeError, TypeError):
            arg_type = 'Real'
        arg_types.append(arg_type)

    # Create the Perl function from the formula
    perlFunction: Callable = _create_perlFunction(formula, arg_names)

    # Create the function definition dictionary
    func_def: Dict[str, Any] = {
        'class': 'parserFunction',
        'argCount': len(arg_names),
        'argNames': arg_names,
        'argTypes': arg_types,
        'function': perlFunction,
        'formula': formula,
        'type': getattr(formula, 'type', 'Real'),
    }

    # Add TeX option for single-letter function names
    if len(func_name) == 1:
        func_def['TeX'] = func_name

    # Merge in any additional kwargs
    func_def.update(kwargs)

    # Add the function to the context
    try:
        context.functions.add(func_name, func_def)
    except (AttributeError, TypeError):
        # Functions might not support add method, store in dict-like manner
        if isinstance(context.functions, dict):
            context.functions[func_name] = func_def


def _create_perlFunction(formula: Any, arg_names: List[str]) -> Callable:
    """
    Create a Python callable from a Formula that evaluates with given arguments.

    Args:
        formula: The Formula object to evaluate
        arg_names: Names of the arguments in order

    Returns:
        A callable that evaluates the formula with the given arguments
    """
    def evaluate(*args: Any, **kwargs: Any) -> Any:
        """
        Evaluate the formula with the given argument values.

        Can be called as:
        - evaluate(value1)           # Single argument by position
        - evaluate(value1, value2)   # Multiple arguments by position
        - evaluate(x=value1)         # Named arguments
        - evaluate(x=value1, y=value2)
        """
        # Build a substitution dictionary
        subs: Dict[str, Any] = {}

        # Handle positional arguments
        for i, arg_value in enumerate(args):
            if i < len(arg_names):
                subs[arg_names[i]] = arg_value

        # Handle keyword arguments
        subs.update(kwargs)

        # Evaluate the formula with substitutions
        if hasattr(formula, 'eval'):
            return formula.eval(**subs)
        elif hasattr(formula, 'substitute'):
            # Some Formula implementations use substitute
            result = formula
            for var_name, var_value in subs.items():
                result = result.substitute(**{var_name: var_value})
            return result
        else:
            # Fallback: try direct evaluation
            return formula(**subs) if callable(formula) else formula

    return evaluate


__all__ = [
    'parserFunction',
]
