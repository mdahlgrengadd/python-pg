"""
PGML Tokenizer - Lexical analysis for PGML syntax.

Recognizes PGML patterns:
- [$var] - Variable interpolation
- [_____] - Answer blank (underscores)
- [@ code @] - Code execution
- [```math```] - Math display block
- [``latex``] - Inline LaTeX
- [* item] - List item
- Bold: *text*, **text**
- Italic: _text_, __text__

Reference: PGML.pl tokenization logic (lines 500-800)
"""

import re
from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """PGML token types (EXPANDED FOR PARITY)."""

    # Text and whitespace
    TEXT = auto()
    NEWLINE = auto()
    BLANK_LINE = auto()
    INDENT = auto()

    # Variable interpolation
    VAR_START = auto()  # [$
    VAR_END = auto()  # ]

    # Answer blanks
    ANSWER_BLANK = auto()  # [_____]
    ANSWER_RULE = auto()  # [@ ans_rule(20) @]

    # Code execution
    CODE_START = auto()  # [@
    CODE_END = auto()  # @]

    # Math display
    MATH_BLOCK_START = auto()  # [```
    MATH_BLOCK_END = auto()  # ```]
    MATH_INLINE_START = auto()  # [``
    MATH_INLINE_END = auto()  # ``]

    # Formatting
    BOLD_START = auto()  # * or **
    BOLD_END = auto()
    ITALIC_START = auto()  # _ or __
    ITALIC_END = auto()

    # Lists
    LIST_ITEM = auto()  # [* item]
    ORDERED_ITEM = auto()  # [1. item]

    # NEW FOR PARITY: Block structures
    HEADING = auto()  # # Heading, ## Subheading, etc.
    RULE = auto()  # --- or ===
    ALIGN_LEFT = auto()  # <<
    ALIGN_RIGHT = auto()  # >>
    ALIGN_CENTER = auto()  # >> ... <<
    PRE_BLOCK = auto()  # :   (indented pre-formatted)

    # NEW FOR PARITY: Tables
    TABLE_ROW_START = auto()  # |
    TABLE_CELL_SEP = auto()  # |
    TABLE_ROW_END = auto()  # |

    # NEW FOR PARITY: Solutions and hints
    SOLUTION_START = auto()  # BEGIN_PGML_SOLUTION
    SOLUTION_END = auto()  # END_PGML_SOLUTION
    HINT_START = auto()  # BEGIN_PGML_HINT
    HINT_END = auto()  # END_PGML_HINT

    # Special
    EOF = auto()


@dataclass
class Token:
    """A PGML token."""

    type: TokenType
    value: str
    line: int
    column: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"


class PGMLTokenizer:
    """
    Tokenize PGML markup into a stream of tokens.

    PGML uses bracket notation for special constructs:
    - [$var] for variables
    - [_____] for answer blanks
    - [@ code @] for Perl code execution
    - [```math```] for display math
    - [``latex``] for inline math
    """

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: list[Token] = []

    def tokenize(self) -> list[Token]:
        """Tokenize the entire input text."""
        while self.pos < len(self.text):
            self._scan_token()

        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, "", self.line, self.column))
        return self.tokens

    def _scan_token(self) -> None:
        """Scan and emit the next token."""
        start_pos = self.pos
        start_line = self.line
        start_col = self.column

        # Check for newlines
        if self._match("\n"):
            # Check if this is a blank line (previous was also newline)
            if self.tokens and self.tokens[-1].type == TokenType.NEWLINE:
                self.tokens[-1] = Token(TokenType.BLANK_LINE,
                                        "\n\n", start_line - 1, 1)
            else:
                self._add_token(TokenType.NEWLINE, "\n", start_line, start_col)
            return

        # NEW FOR PARITY: Check for line-start patterns (headings, rules, tables)
        if self.column == 1 or (self.tokens and self.tokens[-1].type in (TokenType.NEWLINE, TokenType.BLANK_LINE)):
            # Solution/Hint sections
            if self._peek_ahead("BEGIN_PGML_SOLUTION"):
                for _ in range(19):  # len("BEGIN_PGML_SOLUTION")
                    self._advance()
                self._add_token(TokenType.SOLUTION_START,
                                "BEGIN_PGML_SOLUTION", start_line, start_col)
                return

            if self._peek_ahead("END_PGML_SOLUTION"):
                for _ in range(17):  # len("END_PGML_SOLUTION")
                    self._advance()
                self._add_token(TokenType.SOLUTION_END,
                                "END_PGML_SOLUTION", start_line, start_col)
                return

            if self._peek_ahead("BEGIN_PGML_HINT"):
                for _ in range(15):  # len("BEGIN_PGML_HINT")
                    self._advance()
                self._add_token(TokenType.HINT_START,
                                "BEGIN_PGML_HINT", start_line, start_col)
                return

            if self._peek_ahead("END_PGML_HINT"):
                for _ in range(13):  # len("END_PGML_HINT")
                    self._advance()
                self._add_token(TokenType.HINT_END,
                                "END_PGML_HINT", start_line, start_col)
                return

            # Heading: # Header, ## Subheader, etc.
            if self._peek() == "#":
                self._scan_heading()
                return

            # Rule: --- or ===
            if self._peek() in ("-", "="):
                if self._scan_rule():
                    return

            # Table row: | cell | cell |
            if self._peek() == "|":
                self._scan_table_row()
                return

            # Pre-formatted block: :   (colon + spaces)
            if self._peek() == ":" and self._peek_ahead("   "):
                self._scan_pre_block()
                return

        # NEW FOR PARITY: Check for alignment markers
        if self._peek() == ">":
            if self._peek_ahead(">>"):
                self._advance()
                self._advance()
                self._add_token(TokenType.ALIGN_RIGHT,
                                ">>", start_line, start_col)
                return

        if self._peek() == "<":
            if self._peek_ahead("<<"):
                self._advance()
                self._advance()
                self._add_token(TokenType.ALIGN_LEFT, "<<",
                                start_line, start_col)
                return

        # Check for bracket constructs
        if self._peek() == "[":
            self._scan_bracket_construct()
            return

        # Emphasis is not commonly used in PGML - bracket notation is preferred
        # Disable emphasis scanning to avoid conflicts with underscores in text
        # (If needed later, can be re-enabled with proper context checking)

        # Regular text
        self._scan_text()

    def _scan_bracket_construct(self) -> None:
        """Scan bracket-based constructs like [$var], [@code@], etc."""
        start_line = self.line
        start_col = self.column

        if not self._match("["):
            return

        # Look ahead to determine type
        next_char = self._peek()

        # Variable interpolation: [$var]
        if next_char == "$":
            self._advance()  # consume $
            self._add_token(TokenType.VAR_START, "[$", start_line, start_col)
            # Scan variable name
            var_name = self._scan_until("]")
            self._add_token(TokenType.TEXT, var_name, self.line,
                            self.column - len(var_name))
            if self._match("]"):
                self._add_token(TokenType.VAR_END, "]",
                                self.line, self.column - 1)
            return

        # Code execution: [@code@] or [@code@]*
        if next_char == "@":
            self._advance()  # consume @
            self._add_token(TokenType.CODE_START, "[@", start_line, start_col)
            # Scan code until @]
            code = self._scan_until("@]")
            self._add_token(TokenType.TEXT, code, self.line,
                            self.column - len(code))
            if self._match("@]"):
                # Check for trailing * (means display result)
                display_marker = ""
                if self._peek() == "*":
                    display_marker = self._advance()
                self._add_token(
                    TokenType.CODE_END, f"@]{display_marker}", self.line, self.column - len(f"@]{display_marker}"))
            return

        # Math display block: [```...```]
        if self._peek_ahead("```"):
            self._advance()  # consume `
            self._advance()  # consume `
            self._advance()  # consume `
            self._add_token(TokenType.MATH_BLOCK_START,
                            "[```", start_line, start_col)
            # Scan math until ```]
            math = self._scan_until("```]")
            self._add_token(TokenType.TEXT, math, self.line,
                            self.column - len(math))
            if self._match("```]"):
                self._add_token(TokenType.MATH_BLOCK_END,
                                "```]", self.line, self.column - 4)
            return

        # Inline math: [``...``]
        if self._peek_ahead("``"):
            self._advance()  # consume `
            self._advance()  # consume `
            self._add_token(TokenType.MATH_INLINE_START,
                            "[``", start_line, start_col)
            # Scan math until ``]
            math = self._scan_until("``]")
            self._add_token(TokenType.TEXT, math, self.line,
                            self.column - len(math))
            if self._match("``]"):
                self._add_token(TokenType.MATH_INLINE_END,
                                "``]", self.line, self.column - 3)
            return

        # Answer blank: [_____]
        if next_char == "_":
            underscores = ""
            while self._peek() == "_":
                underscores += self._advance()
            if self._match("]"):
                self._add_token(
                    TokenType.ANSWER_BLANK,
                    f"[{underscores}]",
                    start_line,
                    start_col,
                )
            return

        # List item: [* item (ends at newline, no closing bracket)
        if next_char == "*":
            self._advance()  # consume *
            # Scan until newline (list items don't have closing bracket)
            content = self._scan_until("\n")
            self._add_token(TokenType.LIST_ITEM,
                            f"[*{content}", start_line, start_col)
            return

        # NEW FOR PARITY: Check for special markers inside brackets
        # [!text!] - emphasized text (not commonly used, but supported)
        # [::marker::] ... [:::marker] - custom blocks

        if next_char.isdigit():
            # Check for ordered list [1. ... (ends at newline)
            num = ""
            while self._peek().isdigit():
                num += self._advance()
            if self._peek() == ".":
                self._advance()  # consume .
                content = self._scan_until("\n")
                self._add_token(
                    TokenType.ORDERED_ITEM,
                    f"[{num}.{content}",
                    start_line,
                    start_col,
                )
                return

        # Not a recognized PGML bracket construct, treat opening bracket as plain text
        # Back up and consume the [ as ordinary text
        self.pos -= 1
        self.column -= 1
        text = self._advance()  # Consume the [
        # Continue scanning text after the [
        while self.pos < len(self.text):
            char = self._peek()
            if char in ("\n", "["):
                break
            if char == "<" and self._peek_ahead("<<"):
                break
            if char == ">" and self._peek_ahead(">>"):
                break
            text += self._advance()
        if text:
            self._add_token(TokenType.TEXT, text, start_line, start_col)

    def _scan_emphasis(self) -> None:
        """Scan emphasis markers (* or _)."""
        start_line = self.line
        start_col = self.column
        char = self._peek()

        # Check for double emphasis
        if self._peek_ahead(char + char):
            self._advance()
            self._advance()
            token_type = TokenType.BOLD_START if char == "*" else TokenType.ITALIC_START
            self._add_token(token_type, char + char, start_line, start_col)
        else:
            self._advance()
            token_type = TokenType.BOLD_START if char == "*" else TokenType.ITALIC_START
            self._add_token(token_type, char, start_line, start_col)

    def _scan_text(self) -> None:
        """Scan regular text until special character."""
        start_line = self.line
        start_col = self.column
        text = ""

        while self.pos < len(self.text):
            char = self._peek()

            # Stop at newlines and brackets
            if char in ("\n", "["):
                break

            # For < and >, only stop if they're part of << or >>
            if char == "<":
                if self._peek_ahead("<<"):
                    break
                # Otherwise consume it as text
            elif char == ">":
                if self._peek_ahead(">>"):
                    break
                # Otherwise consume it as text

            text += self._advance()

        if text:
            self._add_token(TokenType.TEXT, text, start_line, start_col)

    # NEW FOR PARITY: Scanning methods for new token types

    def _scan_heading(self) -> None:
        """Scan heading: # Heading, ## Subheading, etc."""
        start_line = self.line
        start_col = self.column

        level = 0
        while self._peek() == "#" and level < 6:
            self._advance()
            level += 1

        # Skip optional space after #
        if self._peek() == " ":
            self._advance()

        # Scan heading text until newline
        heading_text = self._scan_until("\n")

        self._add_token(TokenType.HEADING,
                        f"{'#' * level} {heading_text}", start_line, start_col)

    def _scan_rule(self) -> bool:
        """Scan horizontal rule: --- or ===. Returns True if rule found."""
        start_line = self.line
        start_col = self.column

        ch = self._peek()
        if ch not in ("-", "="):
            return False

        rule_chars = ""
        while self._peek() == ch:
            rule_chars += self._advance()

        # Rule requires at least 3 characters
        if len(rule_chars) >= 3:
            self._add_token(TokenType.RULE, rule_chars, start_line, start_col)
            return True

        # Not a rule, backtrack by adding as text
        self._add_token(TokenType.TEXT, rule_chars, start_line, start_col)
        return False

    def _scan_table_row(self) -> None:
        """Scan table row: | cell1 | cell2 | cell3 |"""
        start_line = self.line
        start_col = self.column

        self._advance()  # consume leading |
        self._add_token(TokenType.TABLE_ROW_START, "|", start_line, start_col)

        # Scan cells until end of line
        while self.pos < len(self.text) and self._peek() != "\n":
            # Check for cell separator or end first
            if self._peek() == "|":
                sep_line = self.line
                sep_col = self.column
                self._advance()

                # Check if this is the last | (end of row)
                if self._peek() == "\n" or self.pos >= len(self.text):
                    self._add_token(TokenType.TABLE_ROW_END,
                                    "|", sep_line, sep_col)
                else:
                    self._add_token(TokenType.TABLE_CELL_SEP,
                                    "|", sep_line, sep_col)

            # Check for bracket constructs in cell content
            elif self._peek() == "[":
                self._scan_bracket_construct()

            # Regular text in cell
            else:
                # Scan until | or newline or [
                cell_content = ""
                while self._peek() not in ("|", "\n", "[") and self.pos < len(self.text):
                    cell_content += self._advance()

                if cell_content.strip():
                    self._add_token(TokenType.TEXT, cell_content,
                                    self.line, self.column - len(cell_content))

    def _scan_pre_block(self) -> None:
        """Scan pre-formatted block: :   followed by content."""
        start_line = self.line
        start_col = self.column

        # Consume :   (colon + 3 spaces)
        self._advance()  # :
        for _ in range(3):
            if self._peek() == " ":
                self._advance()

        # Scan content until newline
        content = self._scan_until("\n")

        self._add_token(TokenType.PRE_BLOCK,
                        f":   {content}", start_line, start_col)

    def _scan_until(self, delimiter: str) -> str:
        """Scan text until delimiter is found."""
        text = ""
        while self.pos < len(self.text):
            if self._peek_ahead(delimiter):
                break
            text += self._advance()
        return text

    def _peek(self) -> str:
        """Peek at current character without consuming."""
        if self.pos >= len(self.text):
            return ""
        return self.text[self.pos]

    def _peek_ahead(self, s: str) -> bool:
        """Check if upcoming text matches string s."""
        return self.text[self.pos: self.pos + len(s)] == s

    def _match(self, s: str) -> bool:
        """Try to match and consume string s."""
        if self._peek_ahead(s):
            for _ in s:
                self._advance()
            return True
        return False

    def _advance(self) -> str:
        """Consume and return current character."""
        if self.pos >= len(self.text):
            return ""

        char = self.text[self.pos]
        self.pos += 1

        if char == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1

        return char

    def _add_token(self, token_type: TokenType, value: str, line: int, column: int) -> None:
        """Add a token to the list."""
        self.tokens.append(Token(token_type, value, line, column))
