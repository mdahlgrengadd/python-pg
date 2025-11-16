"""
Abstract Syntax Tree (AST) node definitions for mathematical expressions.

This module defines the AST node hierarchy used to represent parsed mathematical
expressions. It follows the Visitor pattern for extensibility.

Reference: lib/Parser.pm and lib/Parser/*.pm in legacy Perl codebase
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol


class ASTVisitor(Protocol):
    """
    Visitor protocol for traversing AST nodes.

    Implementations can provide eval, string rendering, TeX rendering, etc.
    """

    def visit_number(self, node: "Number") -> Any:
        ...

    def visit_variable(self, node: "Variable") -> Any:
        ...

    def visit_constant(self, node: "Constant") -> Any:
        ...

    def visit_string(self, node: "String") -> Any:
        ...

    def visit_binary_op(self, node: "BinaryOp") -> Any:
        ...

    def visit_unary_op(self, node: "UnaryOp") -> Any:
        ...

    def visit_function_call(self, node: "FunctionCall") -> Any:
        ...

    def visit_list(self, node: "List") -> Any:
        ...

    def visit_point(self, node: "Point") -> Any:
        ...

    def visit_vector(self, node: "Vector") -> Any:
        ...

    def visit_matrix(self, node: "Matrix") -> Any:
        ...

    def visit_interval(self, node: "Interval") -> Any:
        ...


class ASTNode(ABC):
    """
    Base class for all AST nodes.

    Uses the Visitor pattern to allow multiple operations (eval, string, TeX)
    without modifying node classes.
    """

    @abstractmethod
    def accept(self, visitor: ASTVisitor) -> Any:
        """Accept a visitor for traversal."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """Return string representation for debugging."""
        pass


# Leaf Nodes (terminals)


class Number(ASTNode):
    """
    Represents a numeric literal.

    Examples: 42, 3.14, -2.5, 1e-10
    """

    def __init__(self, value: float | int):
        self.value = float(value)

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_number(self)

    def __repr__(self) -> str:
        return f"Number({self.value})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Number) and self.value == other.value


class Variable(ASTNode):
    """
    Represents a variable.

    Examples: x, y, theta, alpha
    """

    def __init__(self, name: str):
        self.name = name

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_variable(self)

    def __repr__(self) -> str:
        return f"Variable('{self.name}')"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Variable) and self.name == other.name


class Constant(ASTNode):
    """
    Represents a mathematical constant.

    Examples: pi, e, i (imaginary unit), inf
    """

    def __init__(self, name: str):
        self.name = name

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_constant(self)

    def __repr__(self) -> str:
        return f"Constant('{self.name}')"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Constant) and self.name == other.name


class String(ASTNode):
    """
    Represents a string literal.

    Examples: "hello", 'world'
    """

    def __init__(self, value: str):
        self.value = value

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_string(self)

    def __repr__(self) -> str:
        return f"String('{self.value}')"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, String) and self.value == other.value


# Composite Nodes (operators and functions)


class BinaryOp(ASTNode):
    """
    Represents a binary operation.

    Examples: 2 + 3, x * y, a ^ b

    Operators: +, -, *, /, ^, **, <, >, <=, >=, ==, !=, &&, ||
    """

    def __init__(self, left: ASTNode, op: str, right: ASTNode):
        self.left = left
        self.op = op
        self.right = right

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_binary_op(self)

    def __repr__(self) -> str:
        return f"BinaryOp({self.left!r}, '{self.op}', {self.right!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BinaryOp)
            and self.left == other.left
            and self.op == other.op
            and self.right == other.right
        )


class UnaryOp(ASTNode):
    """
    Represents a unary operation.

    Examples: -x, !n, +5

    Operators: -, +, !, ~
    """

    def __init__(self, op: str, operand: ASTNode):
        self.op = op
        self.operand = operand

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_unary_op(self)

    def __repr__(self) -> str:
        return f"UnaryOp('{self.op}', {self.operand!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, UnaryOp)
            and self.op == other.op
            and self.operand == other.operand
        )


class FunctionCall(ASTNode):
    """
    Represents a function call.

    Examples: sin(x), sqrt(2), gcd(12, 18), max(a, b, c)
    """

    def __init__(self, name: str, args: list[ASTNode]):
        self.name = name
        self.args = args

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_function_call(self)

    def __repr__(self) -> str:
        args_repr = ", ".join(repr(arg) for arg in self.args)
        return f"FunctionCall('{self.name}', [{args_repr}])"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FunctionCall)
            and self.name == other.name
            and self.args == other.args
        )


# Higher-order Nodes (collections and mathematical structures)


class List(ASTNode):
    """
    Represents a list/sequence.

    Examples: [1, 2, 3], [x, y, z]
    """

    def __init__(self, elements: list[ASTNode]):
        self.elements = elements

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_list(self)

    def __repr__(self) -> str:
        elements_repr = ", ".join(repr(el) for el in self.elements)
        return f"List([{elements_repr}])"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, List) and self.elements == other.elements


class Point(ASTNode):
    """
    Represents a point in n-dimensional space.

    Examples: (1, 2), (x, y, z)

    Notation: Parentheses with comma separation
    """

    def __init__(self, coords: list[ASTNode]):
        self.coords = coords

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_point(self)

    def __repr__(self) -> str:
        coords_repr = ", ".join(repr(c) for c in self.coords)
        return f"Point([{coords_repr}])"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Point) and self.coords == other.coords


class Vector(ASTNode):
    """
    Represents a vector.

    Examples: <1, 2, 3>, <x, y>

    Notation: Angle brackets with comma separation
    """

    def __init__(self, components: list[ASTNode]):
        self.components = components

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_vector(self)

    def __repr__(self) -> str:
        components_repr = ", ".join(repr(c) for c in self.components)
        return f"Vector([{components_repr}])"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Vector) and self.components == other.components


class Matrix(ASTNode):
    """
    Represents a matrix.

    Examples: [[1, 2], [3, 4]], [[a, b], [c, d]]

    Notation: Nested lists (row-major order)
    """

    def __init__(self, rows: list[list[ASTNode]]):
        self.rows = rows

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_matrix(self)

    def __repr__(self) -> str:
        rows_repr = ", ".join(
            "[" + ", ".join(repr(el) for el in row) + "]" for row in self.rows
        )
        return f"Matrix([{rows_repr}])"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Matrix) and self.rows == other.rows


class Interval(ASTNode):
    """
    Represents an interval.

    Examples: [0, 1], (0, 1), [0, 1), (-inf, inf)

    Notation:
    - [ or ] for closed endpoints
    - ( or ) for open endpoints
    """

    def __init__(
        self, left: ASTNode, right: ASTNode, open_left: bool, open_right: bool
    ):
        self.left = left
        self.right = right
        self.open_left = open_left
        self.open_right = open_right

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_interval(self)

    def __repr__(self) -> str:
        left_bracket = "(" if self.open_left else "["
        right_bracket = ")" if self.open_right else "]"
        return f"Interval({left_bracket}{self.left!r}, {self.right!r}{right_bracket})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Interval)
            and self.left == other.left
            and self.right == other.right
            and self.open_left == other.open_left
            and self.open_right == other.open_right
        )
