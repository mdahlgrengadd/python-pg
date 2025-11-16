"""
Recursive descent parser for mathematical expressions.

This parser uses operator precedence (Pratt parsing) to build an Abstract Syntax Tree
from a token stream. It handles:
- Binary and unary operators with precedence
- Function calls
- Multiple parenthesis types (for points, vectors, etc.)
- Implicit multiplication

Reference: lib/Parser.pm::parse() (lines 147-171) in legacy Perl codebase
"""

from typing import TYPE_CHECKING

from .ast import (
    ASTNode,
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
from .tokenizer import Token, TokenType, Tokenizer

if TYPE_CHECKING:
    pass


class ParseError(Exception):
    """Exception raised during parsing."""

    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(f"{message} at position {token.pos}: '{token.value}'")


class Parser:
    """
    Recursive descent parser with operator precedence (Pratt parsing).

    The parser builds an AST from a token stream, respecting:
    - Operator precedence (defined in context)
    - Operator associativity (left/right)
    - Function calls
    - Various bracket types (parentheses, vectors, intervals)
    """

    def __init__(self, context: Context | None = None):
        """
        Initialize parser with optional context.

        Args:
            context: Mathematical context (defaults to Numeric)
        """
        self.context = context or Context.numeric()
        self.tokens: list[Token] = []
        self.pos = 0

    def parse(self, expression: str) -> ASTNode:
        """
        Parse an expression string to an AST.

        Args:
            expression: The mathematical expression

        Returns:
            Root AST node

        Raises:
            ParseError: If expression is invalid
        """
        # Tokenize
        tokenizer = Tokenizer(self.context)
        self.tokens = tokenizer.tokenize(expression)
        self.pos = 0

        # Parse
        if len(self.tokens) == 1 and self.tokens[0].type == TokenType.EOF:
            raise ParseError("Empty expression", self.tokens[0])

        ast = self.parse_expression(0)

        # Ensure we consumed all tokens (except EOF)
        if self.current().type != TokenType.EOF:
            raise ParseError("Unexpected token", self.current())

        return ast

    def current(self) -> Token:
        """Get current token without consuming it."""
        return self.tokens[self.pos]

    def peek(self, offset: int = 1) -> Token:
        """Look ahead at token at offset from current position."""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return self.tokens[-1]  # EOF

    def advance(self) -> Token:
        """Consume and return current token."""
        token = self.current()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token

    def expect(self, token_type: TokenType) -> Token:
        """
        Consume a token of the expected type.

        Args:
            token_type: Expected token type

        Returns:
            The consumed token

        Raises:
            ParseError: If current token doesn't match expected type
        """
        token = self.current()
        if token.type != token_type:
            raise ParseError(
                f"Expected {token_type.name}, got {token.type.name}", token
            )
        return self.advance()

    def parse_expression(self, min_precedence: int = 0) -> ASTNode:
        """
        Parse an expression using operator precedence climbing.

        This implements Pratt parsing (operator precedence parsing).

        Args:
            min_precedence: Minimum precedence to consider

        Returns:
            AST node
        """
        # Parse left side (prefix)
        left = self.parse_prefix()

        # Parse infix operators
        while True:
            token = self.current()

            # Check if this is an infix operator
            if not self.is_binary_operator(token):
                break

            # Get operator precedence
            precedence = self.context.get_operator_precedence(token.value)

            # Stop if precedence is too low
            if precedence < min_precedence:
                break

            # Consume operator
            op_token = self.advance()

            # Determine next min precedence based on associativity
            from .context import Associativity

            assoc = self.context.get_operator_associativity(op_token.value)
            next_min_prec = precedence + (1 if assoc == Associativity.LEFT else 0)

            # Parse right side
            right = self.parse_expression(next_min_prec)

            # Build binary operation node
            left = BinaryOp(left, op_token.value, right)

        return left

    def parse_prefix(self) -> ASTNode:
        """
        Parse a prefix expression (unary operators, atoms, function calls).

        Returns:
            AST node
        """
        token = self.current()

        # Unary operators (-, +, !)
        if token.type in (TokenType.MINUS, TokenType.PLUS, TokenType.NOT):
            op_token = self.advance()
            operand = self.parse_prefix()  # Right associative
            return UnaryOp(op_token.value, operand)

        # Atoms and postfix expressions
        return self.parse_postfix()

    def parse_postfix(self) -> ASTNode:
        """
        Parse postfix expressions (function calls, indexing).

        Returns:
            AST node
        """
        atom = self.parse_atom()

        # Currently no postfix operators in basic math
        # Could add array indexing, factorial, etc. here

        return atom

    def parse_atom(self) -> ASTNode:
        """
        Parse an atomic expression (number, variable, parentheses, etc.).

        Returns:
            AST node
        """
        token = self.current()

        # Number
        if token.type == TokenType.NUMBER:
            self.advance()
            return Number(float(token.value))

        # String
        if token.type == TokenType.STRING:
            self.advance()
            return String(token.value)

        # Constant
        if token.type == TokenType.CONSTANT:
            self.advance()
            return Constant(token.value)

        # Variable or Function
        if token.type == TokenType.VARIABLE:
            self.advance()
            return Variable(token.value)

        # Function call
        if token.type == TokenType.FUNCTION:
            return self.parse_function_call()

        # Parentheses: (expr) or point (x, y)
        if token.type == TokenType.LPAREN:
            return self.parse_parenthesized()

        # Brackets: [expr] or list [1, 2, 3] or interval [0, 1]
        if token.type == TokenType.LBRACKET:
            return self.parse_bracketed()

        # Angle brackets: <expr> or vector <1, 2, 3>
        # Note: < is tokenized as LT (less than), so we need to check for it
        if token.type == TokenType.LANGLE or token.type == TokenType.LT:
            return self.parse_angle_bracketed()

        # Absolute value: |expr|
        if token.type == TokenType.PIPE:
            return self.parse_absolute_value()

        raise ParseError(f"Unexpected token in atom", token)

    def parse_function_call(self) -> FunctionCall:
        """
        Parse a function call: func(arg1, arg2, ...).

        Returns:
            FunctionCall node
        """
        func_token = self.advance()
        func_name = func_token.value

        # Expect opening parenthesis
        self.expect(TokenType.LPAREN)

        # Parse arguments
        args: list[ASTNode] = []

        if self.current().type != TokenType.RPAREN:
            args.append(self.parse_expression())

            while self.current().type == TokenType.COMMA:
                self.advance()  # Consume comma
                args.append(self.parse_expression())

        # Expect closing parenthesis
        self.expect(TokenType.RPAREN)

        return FunctionCall(func_name, args)

    def parse_parenthesized(self) -> ASTNode:
        """
        Parse parenthesized expression: (expr) or point (x, y).

        Returns:
            AST node (potentially wrapped, or Point)
        """
        self.expect(TokenType.LPAREN)

        # Check for empty parentheses
        if self.current().type == TokenType.RPAREN:
            raise ParseError("Empty parentheses", self.current())

        # Parse first expression
        first = self.parse_expression()

        # Check if this is a point (has comma)
        if self.current().type == TokenType.COMMA:
            coords = [first]
            while self.current().type == TokenType.COMMA:
                self.advance()  # Consume comma
                coords.append(self.parse_expression())

            self.expect(TokenType.RPAREN)
            return Point(coords)

        # Just a grouped expression
        self.expect(TokenType.RPAREN)
        return first

    def parse_bracketed(self) -> ASTNode:
        """
        Parse bracketed expression: [expr], list [1, 2], or interval [0, 1].

        Determines type based on context:
        - Single expr: just grouped
        - Multiple exprs with comma: list
        - Two exprs: could be interval (if context is Interval)

        Returns:
            AST node
        """
        open_left = False  # [ means closed
        self.advance()  # Consume [

        # Check for empty brackets
        if self.current().type == TokenType.RBRACKET:
            raise ParseError("Empty brackets", self.current())

        # Parse first expression
        first = self.parse_expression()

        # Check what follows
        if self.current().type == TokenType.COMMA:
            # List or matrix
            elements = [first]

            # Check if first element is itself a list (matrix)
            is_matrix = isinstance(first, List)

            while self.current().type == TokenType.COMMA:
                self.advance()  # Consume comma

                # Allow trailing comma
                if self.current().type == TokenType.RBRACKET:
                    break

                elem = self.parse_expression()
                elements.append(elem)

                # Check consistency for matrix
                if is_matrix and not isinstance(elem, List):
                    raise ParseError("Inconsistent matrix rows", self.current())
                elif not is_matrix and isinstance(elem, List):
                    raise ParseError("Inconsistent matrix rows", self.current())

            self.expect(TokenType.RBRACKET)

            # Return matrix or list
            if is_matrix:
                rows = [
                    [el for el in row.elements] if isinstance(row, List) else [row]
                    for row in elements
                ]
                return Matrix(rows)
            else:
                return List(elements)

        # Single element - check for interval
        # In interval notation: [a, b] or [a, b)
        # We'll handle this if we see closing bracket or closing paren
        if self.current().type == TokenType.RBRACKET:
            # Could be interval [a, b] with two elements separated by comma
            # But we already consumed first element and no comma
            # So this is just [expr] - return it wrapped
            self.advance()  # Consume ]
            return first

        raise ParseError("Expected comma or closing bracket", self.current())

    def parse_angle_bracketed(self) -> Vector:
        """
        Parse angle-bracketed expression: <1, 2, 3> (vector).

        Returns:
            Vector node
        """
        self.advance()  # Consume <

        components: list[ASTNode] = []

        # Parse components
        # > is tokenized as GT, so check for both RANGLE and GT
        if self.current().type not in (TokenType.RANGLE, TokenType.GT):
            # Use precedence 4 to prevent comparison operators from being parsed
            # (comparison operators have precedence 3, so this will stop at >)
            components.append(self.parse_expression(min_precedence=4))

            while self.current().type == TokenType.COMMA:
                self.advance()  # Consume comma
                components.append(self.parse_expression(min_precedence=4))

        # Expect closing >
        # Note: > might be tokenized as GT, need to handle
        if self.current().type == TokenType.RANGLE:
            self.advance()
        elif self.current().type == TokenType.GT:
            self.advance()
        else:
            raise ParseError("Expected closing >", self.current())

        return Vector(components)

    def parse_absolute_value(self) -> FunctionCall:
        """
        Parse absolute value: |expr|.

        Returns:
            FunctionCall node for abs()
        """
        self.advance()  # Consume first |

        expr = self.parse_expression()

        # Expect closing |
        self.expect(TokenType.PIPE)

        return FunctionCall("abs", [expr])

    def is_binary_operator(self, token: Token) -> bool:
        """
        Check if token is a binary operator.

        Args:
            token: Token to check

        Returns:
            True if binary operator
        """
        binary_ops = {
            TokenType.PLUS,
            TokenType.MINUS,
            TokenType.MULTIPLY,
            TokenType.DIVIDE,
            TokenType.POWER,
            TokenType.MODULO,
            TokenType.EQ,
            TokenType.NE,
            TokenType.LT,
            TokenType.LE,
            TokenType.GT,
            TokenType.GE,
            TokenType.AND,
            TokenType.OR,
        }

        return token.type in binary_ops
