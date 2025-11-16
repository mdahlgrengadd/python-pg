"""
AST Visitor implementations for various operations.

Visitors implement the Visitor pattern to traverse and operate on AST nodes:
- StringVisitor: Convert AST to string representation
- TeXVisitor: Convert AST to LaTeX representation
- EvalVisitor: Evaluate AST to numeric values (requires MathObjects)

Reference: lib/Parser.pm evaluation and rendering methods
"""

import math
from typing import Any

from .ast import (
    BinaryOp,
    Constant,
    FunctionCall,
    Interval,
    List,
    Matrix,
    Number,
    Point,
    String,
    UnaryOp,
    Variable,
    Vector,
)
from .context import Context


class StringVisitor:
    """
    Convert AST to string representation.

    Examples:
    - BinaryOp(Number(2), '+', Number(3)) → "2 + 3"
    - FunctionCall('sin', [Variable('x')]) → "sin(x)"
    """

    def __init__(self, context: Context | None = None):
        self.context = context or Context.numeric()

    def visit_number(self, node: Number) -> str:
        # Format number nicely (remove .0 for integers)
        if node.value == int(node.value):
            return str(int(node.value))
        return str(node.value)

    def visit_variable(self, node: Variable) -> str:
        return node.name

    def visit_constant(self, node: Constant) -> str:
        return node.name

    def visit_string(self, node: String) -> str:
        return f'"{node.value}"'

    def visit_binary_op(self, node: BinaryOp) -> str:
        left_str = node.left.accept(self)
        right_str = node.right.accept(self)

        # Add parentheses if needed based on precedence
        left_prec = self._get_precedence(node.left)
        right_prec = self._get_precedence(node.right)
        op_prec = self.context.get_operator_precedence(node.op)

        if left_prec > 0 and left_prec < op_prec:
            left_str = f"({left_str})"

        if right_prec > 0 and right_prec <= op_prec:
            right_str = f"({right_str})"

        return f"{left_str} {node.op} {right_str}"

    def visit_unary_op(self, node: UnaryOp) -> str:
        operand_str = node.operand.accept(self)

        # Add parentheses for complex expressions
        if isinstance(node.operand, BinaryOp):
            operand_str = f"({operand_str})"

        return f"{node.op}{operand_str}"

    def visit_function_call(self, node: FunctionCall) -> str:
        args_str = ", ".join(arg.accept(self) for arg in node.args)
        return f"{node.name}({args_str})"

    def visit_list(self, node: List) -> str:
        elements_str = ", ".join(el.accept(self) for el in node.elements)
        return f"[{elements_str}]"

    def visit_point(self, node: Point) -> str:
        coords_str = ", ".join(coord.accept(self) for coord in node.coords)
        return f"({coords_str})"

    def visit_vector(self, node: Vector) -> str:
        components_str = ", ".join(comp.accept(self) for comp in node.components)
        return f"<{components_str}>"

    def visit_matrix(self, node: Matrix) -> str:
        rows_str = ", ".join(
            "[" + ", ".join(self.visit_number(Number(el)) if isinstance(el, (int, float)) else el.accept(self) for el in row) + "]"
            for row in node.rows
        )
        return f"[{rows_str}]"

    def visit_interval(self, node: Interval) -> str:
        left_bracket = "(" if node.open_left else "["
        right_bracket = ")" if node.open_right else "]"
        left_str = node.left.accept(self)
        right_str = node.right.accept(self)
        return f"{left_bracket}{left_str}, {right_str}{right_bracket}"

    def _get_precedence(self, node: Any) -> int:
        """Get precedence of a node for parenthesization."""
        if isinstance(node, BinaryOp):
            return self.context.get_operator_precedence(node.op)
        return 0


class TeXVisitor:
    """
    Convert AST to LaTeX representation.

    Examples:
    - BinaryOp(Number(2), '*', Variable('x')) → "2x"
    - FunctionCall('sqrt', [Number(2)]) → "\\sqrt{2}"
    - BinaryOp(Variable('x'), '^', Number(2)) → "x^{2}"
    """

    def __init__(self, context: Context | None = None):
        self.context = context or Context.numeric()

    def visit_number(self, node: Number) -> str:
        # Format number nicely
        if node.value == int(node.value):
            return str(int(node.value))
        return str(node.value)

    def visit_variable(self, node: Variable) -> str:
        # Check if variable has custom LaTeX representation
        if self.context.is_variable(node.name):
            var_config = self.context.variables[node.name]
            if var_config.latex:
                return var_config.latex

        # Greek letters and special variables
        greek = {
            "alpha": r"\alpha",
            "beta": r"\beta",
            "gamma": r"\gamma",
            "delta": r"\delta",
            "epsilon": r"\epsilon",
            "theta": r"\theta",
            "lambda": r"\lambda",
            "mu": r"\mu",
            "pi": r"\pi",
            "sigma": r"\sigma",
            "phi": r"\phi",
            "omega": r"\omega",
        }

        if node.name.lower() in greek:
            return greek[node.name.lower()]

        # Multi-character variables
        if len(node.name) > 1:
            return f"\\mathrm{{{node.name}}}"

        return node.name

    def visit_constant(self, node: Constant) -> str:
        constants_tex = {
            "pi": r"\pi",
            "e": "e",
            "i": "i",
            "inf": r"\infty",
            "infinity": r"\infty",
        }
        return constants_tex.get(node.name.lower(), node.name)

    def visit_string(self, node: String) -> str:
        return f"\\text{{{node.value}}}"

    def visit_binary_op(self, node: BinaryOp) -> str:
        left_str = node.left.accept(self)
        right_str = node.right.accept(self)

        # Special handling for different operators
        if node.op == "*":
            # Implicit multiplication or explicit \cdot
            # Check if we should show multiplication
            # Generally use implicit for number * variable
            if isinstance(node.left, Number) and isinstance(
                node.right, (Variable, Constant)
            ):
                return f"{left_str}{right_str}"
            else:
                return f"{left_str} \\cdot {right_str}"

        elif node.op in ("^", "**"):
            # Power: use ^{} syntax
            return f"{left_str}^{{{right_str}}}"

        elif node.op == "/":
            # Fraction
            return f"\\frac{{{left_str}}}{{{right_str}}}"

        elif node.op in ("+", "-"):
            # Addition/subtraction - may need parentheses
            left_prec = self._get_precedence(node.left)
            op_prec = self.context.get_operator_precedence(node.op)

            if left_prec > 0 and left_prec < op_prec:
                left_str = f"\\left({left_str}\\right)"

            # For right side, check if it's a binary op with lower precedence
            if isinstance(node.right, BinaryOp):
                right_prec = self._get_precedence(node.right)
                if right_prec < op_prec:
                    right_str = f"\\left({right_str}\\right)"

            return f"{left_str} {node.op} {right_str}"

        elif node.op in ("<", ">", "<=", ">=", "==", "!="):
            # Comparison operators
            op_tex = {
                "<": "<",
                ">": ">",
                "<=": r"\le",
                ">=": r"\ge",
                "==": "=",
                "!=": r"\ne",
            }
            return f"{left_str} {op_tex[node.op]} {right_str}"

        else:
            # Default: just use operator as-is
            return f"{left_str} {node.op} {right_str}"

    def visit_unary_op(self, node: UnaryOp) -> str:
        operand_str = node.operand.accept(self)

        # Add parentheses for complex expressions
        if isinstance(node.operand, BinaryOp):
            operand_str = f"\\left({operand_str}\\right)"

        if node.op == "-":
            return f"-{operand_str}"
        elif node.op == "+":
            return f"+{operand_str}"
        else:
            return f"{node.op}{operand_str}"

    def visit_function_call(self, node: FunctionCall) -> str:
        # Special LaTeX handling for common functions
        func_name = node.name

        # Trig and other standard functions
        standard_funcs = {
            "sin",
            "cos",
            "tan",
            "sec",
            "csc",
            "cot",
            "sinh",
            "cosh",
            "tanh",
            "ln",
            "log",
            "exp",
            "arcsin",
            "arccos",
            "arctan",
        }

        if func_name in standard_funcs:
            if len(node.args) == 1:
                arg_str = node.args[0].accept(self)
                # Add parentheses for complex arguments
                if isinstance(node.args[0], BinaryOp):
                    arg_str = f"\\left({arg_str}\\right)"
                return f"\\{func_name} {arg_str}"
            else:
                args_str = ", ".join(arg.accept(self) for arg in node.args)
                return f"\\{func_name}\\left({args_str}\\right)"

        # Special functions
        if func_name == "sqrt":
            return f"\\sqrt{{{node.args[0].accept(self)}}}"

        elif func_name == "abs":
            return f"\\left|{node.args[0].accept(self)}\\right|"

        elif func_name in ("frac", "dfrac"):
            return f"\\frac{{{node.args[0].accept(self)}}}{{{node.args[1].accept(self)}}}"

        else:
            # Generic function
            args_str = ", ".join(arg.accept(self) for arg in node.args)
            return f"\\mathrm{{{func_name}}}\\left({args_str}\\right)"

    def visit_list(self, node: List) -> str:
        elements_str = ", ".join(el.accept(self) for el in node.elements)
        return f"\\left[{elements_str}\\right]"

    def visit_point(self, node: Point) -> str:
        coords_str = ", ".join(coord.accept(self) for coord in node.coords)
        return f"\\left({coords_str}\\right)"

    def visit_vector(self, node: Vector) -> str:
        components_str = ", ".join(comp.accept(self) for comp in node.components)
        return f"\\left\\langle {components_str} \\right\\rangle"

    def visit_matrix(self, node: Matrix) -> str:
        # Use pmatrix environment
        rows_tex = " \\\\ ".join(
            " & ".join(
                self.visit_number(Number(el)) if isinstance(el, (int, float)) else el.accept(self)
                for el in row
            )
            for row in node.rows
        )
        return f"\\begin{{pmatrix}} {rows_tex} \\end{{pmatrix}}"

    def visit_interval(self, node: Interval) -> str:
        left_bracket = "(" if node.open_left else "["
        right_bracket = ")" if node.open_right else "]"
        left_str = node.left.accept(self)
        right_str = node.right.accept(self)
        return f"\\left{left_bracket}{left_str}, {right_str}\\right{right_bracket}"

    def _get_precedence(self, node: Any) -> int:
        """Get precedence of a node for parenthesization."""
        if isinstance(node, BinaryOp):
            return self.context.get_operator_precedence(node.op)
        return 0


class EvalVisitor:
    """
    Evaluate AST to numeric values.

    This is a simple evaluator that works with Python's built-in math.
    For full MathObjects support, this will be replaced with a more
    sophisticated evaluator in Phase 2.

    Args:
        bindings: Variable name → value mappings
        context: Mathematical context
    """

    def __init__(self, bindings: dict[str, float] | None = None, context: Context | None = None):
        self.bindings = bindings or {}
        self.context = context or Context.numeric()

    def visit_number(self, node: Number) -> float:
        return node.value

    def visit_variable(self, node: Variable) -> float:
        if node.name in self.bindings:
            return self.bindings[node.name]
        raise ValueError(f"Undefined variable: {node.name}")

    def visit_constant(self, node: Constant) -> float | complex:
        if self.context.is_constant(node.name):
            return self.context.get_constant_value(node.name)

        # Fallback constants
        constants = {
            "pi": math.pi,
            "e": math.e,
            "inf": float("inf"),
            "infinity": float("inf"),
        }

        if node.name.lower() in constants:
            return constants[node.name.lower()]

        raise ValueError(f"Undefined constant: {node.name}")

    def visit_string(self, node: String) -> str:
        return node.value

    def visit_binary_op(self, node: BinaryOp) -> float:
        left = node.left.accept(self)
        right = node.right.accept(self)

        if node.op == "+":
            return left + right
        elif node.op == "-":
            return left - right
        elif node.op == "*":
            return left * right
        elif node.op == "/":
            return left / right
        elif node.op in ("^", "**"):
            return left**right
        elif node.op == "%":
            return left % right
        elif node.op == "<":
            return float(left < right)
        elif node.op == ">":
            return float(left > right)
        elif node.op == "<=":
            return float(left <= right)
        elif node.op == ">=":
            return float(left >= right)
        elif node.op == "==":
            return float(left == right)
        elif node.op == "!=":
            return float(left != right)
        else:
            raise ValueError(f"Unknown operator: {node.op}")

    def visit_unary_op(self, node: UnaryOp) -> float:
        operand = node.operand.accept(self)

        if node.op == "-":
            return -operand
        elif node.op == "+":
            return +operand
        elif node.op == "!":
            return float(not operand)
        else:
            raise ValueError(f"Unknown unary operator: {node.op}")

    def visit_function_call(self, node: FunctionCall) -> float:
        args = [arg.accept(self) for arg in node.args]

        # Heaviside/step function: 0 if x <= 0, 1 if x > 0
        def heaviside(x):
            return 1.0 if x > 0 else 0.0

        # Map function names to math functions
        functions = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "atan2": math.atan2,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "asinh": math.asinh,
            "acosh": math.acosh,
            "atanh": math.atanh,
            "ln": math.log,
            "log": math.log10,
            "log10": math.log10,
            "exp": math.exp,
            "sqrt": math.sqrt,
            "abs": abs,
            "floor": math.floor,
            "ceil": math.ceil,
            "round": round,
            "sign": lambda x: math.copysign(1, x) if x != 0 else 0,
            "max": max,
            "min": min,
            "step": heaviside,
            "u": heaviside,  # Alternative name for step function
        }

        if node.name in functions:
            return functions[node.name](*args)

        raise ValueError(f"Unknown function: {node.name}")

    def visit_list(self, node: List) -> list[float]:
        return [el.accept(self) for el in node.elements]

    def visit_point(self, node: Point) -> tuple[float, ...]:
        return tuple(coord.accept(self) for coord in node.coords)

    def visit_vector(self, node: Vector) -> tuple[float, ...]:
        return tuple(comp.accept(self) for comp in node.components)

    def visit_matrix(self, node: Matrix) -> list[list[float]]:
        return [
            [
                el.accept(self) if hasattr(el, "accept") else el
                for el in row
            ]
            for row in node.rows
        ]

    def visit_interval(self, node: Interval) -> tuple[float, float, bool, bool]:
        left = node.left.accept(self)
        right = node.right.accept(self)
        return (left, right, node.open_left, node.open_right)
