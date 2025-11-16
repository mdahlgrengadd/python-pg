"""
Compute() function for parsing and evaluating expressions.

Port to pg_math from pg.mathobjects for Perl 1:1 parity.
"""

import re
import math
from typing import Union, Any


def Compute(expression: Union[str, int, float], context=None):
    """
    Parse and evaluate a mathematical expression.

    If the expression is constant, returns a Real number.
    If the expression contains variables, returns a Formula.
    In Interval context, parses interval notation like [a, b], (a, b), etc.

    Args:
        expression: Expression to parse
        context: Context to use (None = current)

    Returns:
        Real, Formula, Interval, or other MathObject type

    Examples:
        >>> Compute("2+2")
        Real(4)

        >>> Compute("x^2")
        Formula("x^2")

        >>> Compute("sin(pi/2)")
        Real(1.0)

        >>> Context('Interval'); Compute("[1, 5]")
        Interval('[', 1, 5, ']')
    """
    if context is None:
        from .context import get_current_context
        context = get_current_context()

    # If it's already a number, just return Real
    if isinstance(expression, (int, float)):
        from .numeric import Real
        return Real(expression)

    # Convert to string
    expr_str = str(expression).strip()

    # Try to parse as a simple number
    try:
        value = float(expr_str)
        from .numeric import Real
        return Real(value)
    except ValueError:
        pass

    # Check for interval notation in Interval context
    # Pattern: [a, b], (a, b), [a, b), (a, b]
    if context.name == 'Interval' and _is_interval_notation(expr_str):
        return _parse_interval(expr_str, context)

    # Check for vector notation in Vector contexts (or anything that looks like vector notation)
    # Pattern: <a, b>, <a, b, c>, etc.
    # Try vector notation if it looks like one, regardless of context
    if _is_vector_notation(expr_str):
        try:
            return _parse_vector(expr_str, context)
        except Exception:
            # If vector parsing fails, fall through to other handlers
            pass

    # Check for fraction notation in Fraction contexts
    # Pattern: a/b, a b/c (mixed number)
    if 'Fraction' in context.name and _is_fraction_notation(expr_str):
        return _parse_fraction(expr_str, context)

    # Check for unit expressions in Units context
    if context.name == 'Units' or context.name == 'LimitedUnits':
        unit_obj = _parse_unit_expression(expr_str, context)
        if unit_obj is not None:
            return unit_obj

    # Check for assignment expressions if assignments are enabled
    if context.has_assignment_operator() and '=' in expr_str:
        assignment_obj = _parse_assignment(expr_str, context)
        if assignment_obj is not None:
            return assignment_obj

    # Check if it's a constant expression (no variables)
    if _is_constant_expression(expr_str, context):
        # Evaluate as constant
        try:
            value = _evaluate_constant(expr_str, context)
            from .numeric import Real
            return Real(value)
        except Exception:
            pass

    # Otherwise, return as Formula
    from .formula import Formula

    # Extract variables from the expression string, not just context
    # This ensures the Formula has the correct variables that actually appear in the expression
    import re
    # Find all variable names in the expression (alphanumeric sequences that aren't numbers or functions)
    # Common function names to exclude
    function_names = {'sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'asin', 'acos', 'atan', 
                     'sinh', 'cosh', 'tanh', 'log', 'ln', 'exp', 'sqrt', 'abs', 'sgn', 
                     'step', 'fact', 'pi', 'e', 'E', 'PI'}
    # Find all word-like tokens
    var_pattern = r'\b([a-zA-Z][a-zA-Z0-9_]*)\b'
    found_vars = set(re.findall(var_pattern, expr_str))
    # Filter out function names and numbers
    variables = [v for v in found_vars if v not in function_names and not v.replace('_', '').isdigit()]
    
    # If no variables found, use context variables as fallback
    if not variables:
        variables = context.variables.list()

    return Formula(expr_str, variables, context)


def _is_constant_expression(expr: str, context) -> bool:
    """Check if expression contains only constants (no variables)."""
    # Get list of variables from context
    variables = context.variables.list()

    # Remove constants, numbers, operators, functions, and parentheses
    stripped = expr

    # Remove function calls
    for func in context.functions.list():
        stripped = re.sub(rf'\b{func}\s*\(', '', stripped)

    # Remove constants
    for const in context.constants.list():
        stripped = stripped.replace(const, '')

    # Remove numbers (including decimals and scientific notation)
    stripped = re.sub(r'\b\d+\.?\d*([eE][+-]?\d+)?\b', '', stripped)

    # Remove operators and parentheses
    for char in '+-*/^()[], \t\n':
        stripped = stripped.replace(char, '')

    # If anything remains, it might be a variable
    if stripped:
        # Check if any remaining parts are variables
        parts = re.findall(r'\b\w+\b', stripped)
        for part in parts:
            if part in variables:
                return False

    return True


def _evaluate_constant(expr: str, context) -> float:
    """
    Evaluate a constant expression.

    This is a simple evaluator for constant expressions.
    Uses Python's eval() with a restricted namespace.
    """
    # Build safe namespace
    namespace = {
        # Constants
        'pi': math.pi,
        'e': math.e,
        # Functions
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'sinh': math.sinh,
        'cosh': math.cosh,
        'tanh': math.tanh,
        'asinh': math.asinh,
        'acosh': math.acosh,
        'atanh': math.atanh,
        'exp': math.exp,
        'ln': math.log,
        'log': math.log,
        'log10': math.log10,
        'sqrt': math.sqrt,
        'abs': abs,
        'int': int,
        # Add context constants
        **{name: context.constants.get(name) for name in context.constants.list()},
    }

    # Convert ^ to **
    expr = expr.replace('^', '**')

    # Evaluate safely
    try:
        result = eval(expr, {"__builtins__": {}}, namespace)
        return float(result)
    except Exception as e:
        raise ValueError(f"Cannot evaluate expression: {expr}") from e


def _is_interval_notation(expr: str) -> bool:
    """
    Check if expression looks like interval notation.

    Patterns: [a, b], (a, b), [a, b), (a, b]
    """
    expr = expr.strip()
    if len(expr) < 5:
        return False

    # Must start with [ or ( and end with ] or )
    if expr[0] not in ['[', '('] or expr[-1] not in [']', ')']:
        return False

    # Must contain a comma
    if ',' not in expr:
        return False

    # Extract content inside brackets
    content = expr[1:-1].strip()

    # Should have exactly one comma at the top level (not inside nested parentheses)
    depth = 0
    comma_count = 0
    for ch in content:
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        elif ch == ',' and depth == 0:
            comma_count += 1

    return comma_count == 1


def _parse_interval(expr: str, context) -> 'Interval':
    """
    Parse interval notation into an Interval object.

    Syntax: [a, b], (a, b), [a, b), (a, b]
    Endpoints can be numbers or inf/infinity/-inf/-infinity

    Reference: lib/Parser/List/Interval.pm, lib/Value/Interval.pm
    """
    from .sets import Interval

    expr = expr.strip()

    # Extract open/close brackets
    open_bracket = expr[0]
    close_bracket = expr[-1]

    # Extract content
    content = expr[1:-1].strip()

    # Split on comma (at top level only)
    depth = 0
    comma_pos = -1
    for i, ch in enumerate(content):
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        elif ch == ',' and depth == 0:
            comma_pos = i
            break

    if comma_pos < 0:
        raise ValueError(f"Intervals must have two endpoints: {expr}")

    left_str = content[:comma_pos].strip()
    right_str = content[comma_pos + 1:].strip()

    # Parse endpoints
    left_value = _parse_interval_endpoint(left_str, context)
    right_value = _parse_interval_endpoint(right_str, context)

    # Validate according to Perl rules (lib/Parser/List/Interval.pm::_check)

    # Check for positive infinity on left
    if left_value == float('inf'):
        raise ValueError(
            "The left endpoint of an interval can't be positive infinity")

    # Check for negative infinity on right
    if right_value == float('-inf'):
        raise ValueError(
            "The right endpoint of an interval can't be negative infinity")

    # Check that infinite endpoints are open
    if left_value == float('-inf') and open_bracket != '(':
        raise ValueError("Infinite endpoints must be open")

    if right_value == float('inf') and close_bracket != ')':
        raise ValueError("Infinite endpoints must be open")

    # Check ordering (left < right) for finite endpoints
    if left_value != float('-inf') and right_value != float('inf'):
        if left_value >= right_value:
            raise ValueError("Left endpoint must be less than right endpoint")

    # Create Interval object
    return Interval(open_bracket, left_value, right_value, close_bracket)


def _parse_interval_endpoint(s: str, context) -> float:
    """
    Parse an interval endpoint (number or infinity).

    Reference: lib/Value/Interval.pm::new
    """
    s = s.strip()

    # Check for infinity
    if s in ['inf', 'infinity']:
        return float('inf')
    if s in ['-inf', '-infinity']:
        return float('-inf')

    # Try to parse as number
    try:
        return float(s)
    except ValueError:
        pass

    # Try to evaluate as expression (e.g., "2*pi", "sqrt(2)")
    try:
        value = _evaluate_constant(s, context)
        return value
    except Exception:
        raise ValueError(
            f"Interval endpoints must be numbers or infinity: {s}")


def _is_fraction_notation(expr: str) -> bool:
    """
    Check if expression looks like fraction notation.

    Patterns: a/b, a b/c (mixed number), -a/b

    Reference: contextFraction.pl
    """
    import re
    expr = expr.strip()

    # Pattern: optional sign, optional whole number with space, fraction
    # Examples: 1/2, -3/4, 2 1/2, -1 2/3
    pattern = r'^-?\s*\d+\s*/\s*\d+$'  # Simple fraction
    mixed_pattern = r'^-?\s*\d+\s+\d+\s*/\s*\d+$'  # Mixed number

    return bool(re.match(pattern, expr) or re.match(mixed_pattern, expr))


def _parse_fraction(expr: str, context) -> 'Fraction':
    """
    Parse fraction notation into a Fraction object.

    Syntax: 
        - Simple fractions: "3/4", "-3/4"
        - Mixed numbers: "4 1/2", "-1 1/2" (if allowMixedNumbers is set)
        - Whole numbers: "5", "-5"

    Reference: contextFraction.pl
    """
    from .fraction import Fraction
    import re

    expr = expr.strip()

    # Check for mixed number: "4 1/2" or "-4 1/2"
    mixed_pattern = r'^(-?\d+)\s+(\d+)/(\d+)$'
    mixed_match = re.match(mixed_pattern, expr)

    if mixed_match:
        # Mixed number
        whole = int(mixed_match.group(1))
        frac_num = int(mixed_match.group(2))
        frac_den = int(mixed_match.group(3))

        # Check if mixed numbers are allowed
        if not context.flags.get('allowMixedNumbers'):
            raise ValueError("Mixed numbers are not allowed in this context")

        # Validate that fraction part is proper (numerator < denominator)
        if frac_num >= frac_den:
            raise ValueError(
                f"Fraction part of mixed number must be proper: {frac_num}/{frac_den}")

        # Convert to improper fraction: a b/c = (a*c + b)/c
        # For negative mixed numbers: -a b/c = -(a*c + b)/c = (a*c - b)/c when a is negative
        if whole < 0:
            total_num = whole * frac_den - frac_num
        else:
            total_num = whole * frac_den + frac_num

        return Fraction(total_num, frac_den, context)

    # Simple fraction: "a/b"
    simple_pattern = r'^(-?)\s*(\d+)\s*/\s*(\d+)$'
    simple_match = re.match(simple_pattern, expr)

    if simple_match:
        sign = simple_match.group(1)
        num = int(simple_match.group(2))
        den = int(simple_match.group(3))

        if sign == '-':
            num = -num

        # Check strictFractions flag - division only allowed between integers
        if context.flags.get('strictFractions'):
            # Already integers, so this is fine
            pass

        return Fraction(num, den, context)

    raise ValueError(f"Invalid fraction notation: {expr}")


def _is_vector_notation(expr: str) -> bool:
    """
    Check if expression looks like vector notation.

    Patterns: <a>, <a, b>, <a, b, c>, etc.
    Components can be numbers or expressions.
    """
    expr = expr.strip()
    if len(expr) < 3:
        return False

    # Must start with < and end with >
    if expr[0] != '<' or expr[-1] != '>':
        return False

    # Must contain at least one comma (or single component)
    content = expr[1:-1].strip()

    # Empty is not a valid vector
    if not content:
        return False

    return True


def _parse_vector(expr: str, context) -> 'Vector':
    """
    Parse vector notation into a Vector object.

    Syntax: <a>, <a, b>, <a, b, c>, etc.
    Components can be numbers or expressions.

    Reference: lib/Value/Vector.pm
    """
    from .geometric import Vector

    expr = expr.strip()

    # Extract content inside angle brackets
    content = expr[1:-1].strip()

    # Split by comma (accounting for nested parentheses)
    components = []
    current = ""
    depth = 0

    for ch in content:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        elif ch == ',' and depth == 0:
            components.append(current.strip())
            current = ""
            continue
        current += ch

    # Don't forget the last component
    if current.strip():
        components.append(current.strip())

    # Parse each component (may be numbers or expressions)
    parsed_components = []
    for comp in components:
        try:
            # Try to evaluate as a number
            value = float(comp)
            from .numeric import Real
            parsed_components.append(Real(value))
        except ValueError:
            # Try to parse as an expression (Formula)
            try:
                # Recursively compute to handle formulas/expressions
                result = Compute(comp, context)
                parsed_components.append(result)
            except Exception:
                # If all else fails, just use the string as-is
                from .formula import Formula
                parsed_components.append(
                    Formula(comp, context.variables.list(), context))

    return Vector(*parsed_components)


def _parse_unit_expression(expr: str, context) -> Union[object, None]:
    """
    Parse a unit expression like "75 ml", "33 ft/s", or "x^2 ft".

    Returns a NumberWithUnits or FormulaWithUnits object if recognized,
    otherwise returns None to fall through to other parsing methods.

    Args:
        expr: Expression string that may contain units
        context: Units context

    Returns:
        NumberWithUnits or FormulaWithUnits object, or None
    """
    expr = expr.strip()

    # Try to split on last space to separate formula/number from unit
    # This handles:
    # - "75 ml" → number + unit
    # - "(-16 t^2 + 64 t) ft" → formula + unit
    # - "x^2 + 1 m/s" → formula + unit

    # Look for pattern: (expression) unit or expression unit
    # Split by the LAST space to get potential unit
    parts = expr.rsplit(None, 1)  # Split on last whitespace

    if len(parts) != 2:
        return None

    formula_str, unit_str = parts

    # Check if unit_str looks like a unit (letters, possibly with / or ^)
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9/\*\^\-]*$', unit_str):
        return None

    # Check if unit is actually defined in the context
    if not _is_known_unit(unit_str, context):
        return None

    # Try to parse formula_str as a number first
    try:
        number_val = float(formula_str)
        # It's a number! Return NumberWithUnits
        from pg.macros.contexts.context_units import NumberWithUnits
        return NumberWithUnits(number_val, unit_str, context)
    except (ValueError, SyntaxError):
        pass

    # Not a simple number - try as formula
    try:
        from pg.macros.parsers.parser_formula_with_units import FormulaWithUnits
        return FormulaWithUnits(formula_str, unit_str, context=context)
    except Exception:
        pass

    return None


def _is_known_unit(unit_str: str, context) -> bool:
    """
    Check if a unit string is a known unit in the context.

    Args:
        unit_str: Unit string to check (e.g., 'ml', 'ft/s')
        context: Context to check in

    Returns:
        True if unit is known, False otherwise
    """
    # Handle compound units (e.g., "ft/s")
    if '/' in unit_str:
        parts = unit_str.split('/')
        if len(parts) == 2:
            # Check if both parts are known units
            return (_is_simple_unit_known(parts[0].strip(), context) and
                    _is_simple_unit_known(parts[1].strip(), context))

    return _is_simple_unit_known(unit_str, context)


def _is_simple_unit_known(unit_str: str, context) -> bool:
    """Check if a simple (non-compound) unit is known."""
    try:
        from pg.macros.contexts.context_units import UNIT_DEFINITIONS

        # Check all categories
        for category, units in UNIT_DEFINITIONS.items():
            if unit_str in units:
                return True
            # Check aliases
            for unit_name, unit_info in units.items():
                if unit_str in unit_info.get('aliases', []):
                    return True

        return False
    except ImportError:
        return False


def _parse_assignment(expr: str, context) -> Any:
    """
    Parse an assignment expression (e.g., "y = 3x + 1" or "f(x) = x^2").
    
    Reference: macros/parsers/parserAssignment.pl
    
    Args:
        expr: Expression string that may contain an assignment
        context: Context to use for parsing
        
    Returns:
        Formula with type 'Assignment' wrapping AssignmentValue, or None if not an assignment
    """
    expr = expr.strip()
    
    # Check if it contains '=' (assignment operator)
    if '=' not in expr:
        return None
    
    # Split on '=' at top level (not inside parentheses)
    # Find the first '=' that's not inside parentheses
    depth = 0
    equals_pos = -1
    for i, char in enumerate(expr):
        if char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
        elif char == '=' and depth == 0:
            equals_pos = i
            break
    
    if equals_pos < 0:
        return None
    
    # Split into left and right hand sides
    lhs_str = expr[:equals_pos].strip()
    rhs_str = expr[equals_pos + 1:].strip()
    
    if not lhs_str or not rhs_str:
        return None
    
    # Parse LHS - could be variable or function declaration
    is_function = False
    variable_name = None
    params = []
    
    # Check if LHS is a function declaration: f(x) or f(x, y)
    import re
    func_match = re.match(r'^(\w+)\s*\(([^)]*)\)\s*$', lhs_str)
    if func_match:
        is_function = True
        variable_name = func_match.group(1)
        params_str = func_match.group(2).strip()
        if params_str:
            params = [p.strip() for p in params_str.split(',')]
    else:
        # Simple variable assignment
        var_match = re.match(r'^(\w+)\s*$', lhs_str)
        if not var_match:
            # Invalid LHS - not a simple variable
            return None
        variable_name = var_match.group(1)
    
    # Parse RHS using Compute (recursive)
    try:
        rhs_value = Compute(rhs_str, context)
    except Exception:
        # If RHS parsing fails, this isn't a valid assignment
        return None
    
    # Create AssignmentValue
    from pg.macros.parsers.parser_assignment import AssignmentValue
    assignment_value = AssignmentValue(
        variable_name,
        rhs_value,
        is_function=is_function,
        params=params if is_function else None
    )
    
    # Wrap in Formula with type 'Assignment'
    from .formula import Formula
    # Create a Formula that wraps the AssignmentValue
    # We need to avoid parsing the full expression in Formula since it contains '='
    # Instead, create a minimal Formula and set assignment attributes
    # Use the RHS expression for the Formula's expression (without '=')
    formula = Formula(rhs_str, context.variables.list(), context)
    # Mark it as an assignment formula and store assignment value
    formula._assignment_value = assignment_value
    formula._is_assignment = True
    # Override the expression string to include the full assignment
    formula.expression = expr
    
    return formula
