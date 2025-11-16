"""
Tokenizer for mathematical expressions.

This module provides regex-based tokenization with context-aware patterns.
It handles numbers, variables, operators, functions, and various parenthesis types.

Reference: lib/Parser.pm::tokenize() (lines 100-135) in legacy Perl codebase
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context


class TokenType(Enum):
    """Token types for mathematical expressions."""

    # Literals
    NUMBER = auto()
    VARIABLE = auto()
    CONSTANT = auto()
    STRING = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    POWER = auto()
    MODULO = auto()

    # Comparison
    EQ = auto()  # ==
    NE = auto()  # !=
    LT = auto()  # <
    LE = auto()  # <=
    GT = auto()  # >
    GE = auto()  # >=

    # Logical
    AND = auto()  # &&
    OR = auto()  # ||
    NOT = auto()  # !

    # Parentheses and brackets
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    LBRACE = auto()  # {
    RBRACE = auto()  # }
    LANGLE = auto()  # <
    RANGLE = auto()  # >
    PIPE = auto()  # | (for absolute value)

    # Delimiters
    COMMA = auto()
    SEMICOLON = auto()
    COLON = auto()

    # Special
    FUNCTION = auto()  # Function name followed by (
    EOF = auto()
    UNKNOWN = auto()


@dataclass
class Token:
    """
    Represents a single token in the expression.

    Attributes:
        type: The token type
        value: The string value of the token
        pos: Position in the source string (for error reporting)
    """

    type: TokenType
    value: str
    pos: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, '{self.value}', pos={self.pos})"


class Tokenizer:
    """
    Tokenizes mathematical expressions using context-aware regex patterns.

    The tokenizer handles:
    - Numbers (integers, floats, scientific notation)
    - Variables and constants
    - Operators (arithmetic, comparison, logical)
    - Functions
    - Multiple parenthesis types
    - Implicit multiplication (2x, sin x, (x+1)(x-1))
    """

    # Regex patterns for token matching
    PATTERNS = {
        # Numbers: integer, float, scientific notation
        "NUMBER": r"\d+\.?\d*(?:[eE][+-]?\d+)?",
        # Variables: letter followed by letters/digits/underscores
        "VARIABLE": r"[a-zA-Z_][a-zA-Z0-9_]*",
        # Strings: single or double quoted
        "STRING": r'"[^"]*"|\'[^\']*\'',
        # Operators (order matters - check longer operators first)
        "LE": r"<=",
        "GE": r">=",
        "EQ": r"==",
        "NE": r"!=",
        "AND": r"&&",
        "OR": r"\|\|",
        "POWER": r"\*\*|\^",
        "PLUS": r"\+",
        "MINUS": r"-",
        "MULTIPLY": r"\*",
        "DIVIDE": r"/",
        "MODULO": r"%",
        "LT": r"<",
        "GT": r">",
        "NOT": r"!",
        # Delimiters
        "LPAREN": r"\(",
        "RPAREN": r"\)",
        "LBRACKET": r"\[",
        "RBRACKET": r"\]",
        "LBRACE": r"\{",
        "RBRACE": r"\}",
        "PIPE": r"\|",
        "COMMA": r",",
        "SEMICOLON": r";",
        "COLON": r":",
        # Whitespace (to skip)
        "WHITESPACE": r"\s+",
    }

    # Mathematical constants
    CONSTANTS = {"pi", "e", "i", "inf", "infinity"}

    def __init__(self, context: "Context | None" = None):
        """
        Initialize tokenizer with optional context.

        Args:
            context: Mathematical context defining variables, constants, functions
        """
        self.context = context
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for faster matching."""
        # Create combined pattern with named groups
        pattern_parts = []
        for name, pattern in self.PATTERNS.items():
            pattern_parts.append(f"(?P<{name}>{pattern})")

        self.combined_pattern = re.compile("|".join(pattern_parts))

    def tokenize(self, expression: str) -> list[Token]:
        """
        Tokenize a mathematical expression.

        Args:
            expression: The expression to tokenize

        Returns:
            List of tokens

        Raises:
            ValueError: If expression contains invalid characters
        """
        tokens: list[Token] = []
        pos = 0

        while pos < len(expression):
            match = self.combined_pattern.match(expression, pos)

            if not match:
                raise ValueError(
                    f"Invalid character at position {pos}: '{expression[pos]}'"
                )

            # Get the matched group name and value
            kind = match.lastgroup
            value = match.group()
            token_pos = pos
            pos = match.end()

            # Skip whitespace
            if kind == "WHITESPACE":
                continue

            # Determine token type
            if kind == "NUMBER":
                token_type = TokenType.NUMBER
            elif kind == "VARIABLE":
                # Check if it's a constant or function
                if value.lower() in self.CONSTANTS or (
                    self.context and value in self.context.constants
                ):
                    token_type = TokenType.CONSTANT
                elif self._is_function(value, expression, pos):
                    token_type = TokenType.FUNCTION
                else:
                    token_type = TokenType.VARIABLE
            elif kind == "STRING":
                token_type = TokenType.STRING
                # Remove quotes
                value = value[1:-1]
            else:
                # Direct mapping for operators and delimiters
                try:
                    token_type = TokenType[kind]
                except KeyError:
                    token_type = TokenType.UNKNOWN

            tokens.append(Token(token_type, value, token_pos))

        # Add EOF token
        tokens.append(Token(TokenType.EOF, "", len(expression)))

        # Insert implicit multiplication tokens
        tokens = self._insert_implicit_multiplication(tokens)

        return tokens

    def _is_function(self, name: str, expression: str, pos: int) -> bool:
        """
        Check if a variable name is followed by '(' indicating a function call.

        Args:
            name: The potential function name
            expression: Full expression
            pos: Current position (after the name)

        Returns:
            True if this is a function call
        """
        # Skip whitespace
        while pos < len(expression) and expression[pos].isspace():
            pos += 1

        # Check for opening parenthesis
        if pos < len(expression) and expression[pos] == "(":
            return True

        # Also check context for known functions
        if self.context and name in self.context.functions:
            return True

        return False

    def _insert_implicit_multiplication(self, tokens: list[Token]) -> list[Token]:
        """
        Insert implicit multiplication tokens where appropriate.

        Examples:
        - 2x → 2 * x
        - sin x → sin * x (NO - this is sin(x))
        - (x+1)(x-1) → (x+1) * (x-1)
        - 2(x) → 2 * (x)

        Args:
            tokens: Original token list

        Returns:
            Token list with implicit multiplication inserted
        """
        result: list[Token] = []

        for i, token in enumerate(tokens):
            result.append(token)

            # Skip if this is the last token or EOF
            if i >= len(tokens) - 1 or token.type == TokenType.EOF:
                continue

            next_token = tokens[i + 1]

            # Cases where we insert multiplication:
            # 1. number followed by variable/function: 2x, 2sin
            # 2. number followed by opening paren: 2(x)
            # 3. closing paren followed by opening paren: (x+1)(x-1)
            # 4. closing paren followed by number/variable: (x)2, (x)y
            # 5. variable followed by opening paren (not a function): x(y+1)

            should_insert = False

            if token.type == TokenType.NUMBER:
                if next_token.type in (
                    TokenType.VARIABLE,
                    TokenType.CONSTANT,
                    TokenType.FUNCTION,
                    TokenType.LPAREN,
                    TokenType.LBRACKET,
                    TokenType.LANGLE,
                ):
                    should_insert = True

            elif token.type in (TokenType.RPAREN, TokenType.RBRACKET, TokenType.RBRACE, TokenType.RANGLE, TokenType.PIPE):
                if next_token.type in (
                    TokenType.NUMBER,
                    TokenType.VARIABLE,
                    TokenType.CONSTANT,
                    TokenType.LPAREN,
                    TokenType.LBRACKET,
                    TokenType.LANGLE,
                ):
                    should_insert = True

            elif token.type in (TokenType.VARIABLE, TokenType.CONSTANT):
                # Only if next is opening paren and this var is not a function
                if next_token.type == TokenType.LPAREN and token.type != TokenType.FUNCTION:
                    should_insert = True

            if should_insert:
                # Insert multiplication token at the position between tokens
                mult_token = Token(TokenType.MULTIPLY, "*", token.pos + len(token.value))
                result.append(mult_token)

        return result
