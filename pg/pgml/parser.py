"""
PGML Parser - Parse PGML tokens into document structure.

Creates a document tree with:
- Block elements: paragraphs, lists, math blocks
- Inline elements: text, variables, emphasis, math

Reference: PGML.pl parser logic (lines 800-1200)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .tokenizer import PGMLTokenizer, Token, TokenType


# Abstract Syntax Tree Nodes


class PGMLNode(ABC):
    """Base class for PGML AST nodes."""

    @abstractmethod
    def accept(self, visitor: Any) -> Any:
        """Accept a visitor for rendering."""
        pass


# Block-level nodes


@dataclass
class Document(PGMLNode):
    """Root document node containing blocks."""

    blocks: list[PGMLNode] = field(default_factory=list)

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_document(self)


@dataclass
class Paragraph(PGMLNode):
    """Paragraph block containing inline content."""

    content: list[PGMLNode] = field(default_factory=list)

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_paragraph(self)


@dataclass
class MathBlock(PGMLNode):
    """Display math block."""

    content: str

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_math_block(self)


@dataclass
class List(PGMLNode):
    """Unordered or ordered list."""

    items: list[ListItem] = field(default_factory=list)
    ordered: bool = False

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_list(self)


@dataclass
class ListItem(PGMLNode):
    """List item containing inline content."""

    content: list[PGMLNode] = field(default_factory=list)

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_list_item(self)


# Inline nodes


@dataclass
class Text(PGMLNode):
    """Plain text."""

    content: str

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_text(self)


@dataclass
class Variable(PGMLNode):
    """Variable interpolation [$var]."""

    name: str

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_variable(self)


@dataclass
class AnswerBlank(PGMLNode):
    """Answer input blank."""

    width: int = 20  # Number of underscores or explicit width
    name: str = ""  # Optional answer name
    evaluator_code: str = ""  # Code for evaluator (from {$ans} syntax)

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_answer_blank(self)


@dataclass
class Code(PGMLNode):
    """Code execution block [@code@] or [@code@]*."""

    code: str
    display_result: bool = True  # False for [@code@], True for [@code@]*

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_code(self)


@dataclass
class MathInline(PGMLNode):
    """Inline math [``...``]."""

    content: str

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_math_inline(self)


@dataclass
class Bold(PGMLNode):
    """Bold text."""

    content: list[PGMLNode] = field(default_factory=list)

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_bold(self)


@dataclass
class Italic(PGMLNode):
    """Italic text."""

    content: list[PGMLNode] = field(default_factory=list)

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_italic(self)


# NEW FOR PARITY: Additional block-level nodes


@dataclass
class Heading(PGMLNode):
    """Heading with level (1-6)."""

    level: int  # 1 for #, 2 for ##, etc.
    content: list[PGMLNode] = field(default_factory=list)

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_heading(self)


@dataclass
class Rule(PGMLNode):
    """Horizontal rule (--- or ===)."""

    style: str = "-"  # "-" or "="

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_rule(self)


@dataclass
class Table(PGMLNode):
    """Table with rows and cells."""

    rows: list[TableRow] = field(default_factory=list)

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_table(self)


@dataclass
class TableRow(PGMLNode):
    """Table row with cells."""

    # Each cell contains inline content
    cells: list[list[PGMLNode]] = field(default_factory=list)

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_table_row(self)


@dataclass
class AlignBlock(PGMLNode):
    """Alignment block (left, right, center)."""

    alignment: str  # "left", "right", "center"
    content: list[PGMLNode] = field(default_factory=list)

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_align_block(self)


@dataclass
class PreBlock(PGMLNode):
    """Pre-formatted code block."""

    content: str

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_pre_block(self)


@dataclass
class Solution(PGMLNode):
    """Solution section (BEGIN_PGML_SOLUTION...END_PGML_SOLUTION)."""

    content: list[PGMLNode] = field(default_factory=list)

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_solution(self)


@dataclass
class Hint(PGMLNode):
    """Hint section (BEGIN_PGML_HINT...END_PGML_HINT)."""

    content: list[PGMLNode] = field(default_factory=list)

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_hint(self)


# Parser


class PGMLParser:
    """
    Parse PGML tokens into a document tree.

    The parser builds a hierarchical structure:
    - Document (root)
      - Blocks (paragraphs, lists, math blocks)
        - Inline elements (text, variables, emphasis, math)
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[0] if tokens else None

    @classmethod
    def parse_text(cls, text: str) -> Document:
        """Convenience method to tokenize and parse text."""
        tokenizer = PGMLTokenizer(text)
        tokens = tokenizer.tokenize()
        parser = cls(tokens)
        return parser.parse()

    def parse(self) -> Document:
        """Parse tokens into document tree."""
        blocks: list[PGMLNode] = []

        while not self._is_at_end():
            # Skip blank lines between blocks
            while self._match(TokenType.BLANK_LINE, TokenType.NEWLINE):
                pass

            if self._is_at_end():
                break

            # Parse block-level elements
            block = self._parse_block()
            if block:
                blocks.append(block)

        return Document(blocks=blocks)

    def _parse_block(self) -> PGMLNode | None:
        """Parse a block-level element."""
        # Math block
        if self._check(TokenType.MATH_BLOCK_START):
            return self._parse_math_block()

        # List
        if self._check(TokenType.LIST_ITEM) or self._check(TokenType.ORDERED_ITEM):
            return self._parse_list()

        # NEW FOR PARITY: Additional block types

        # Heading
        if self._check(TokenType.HEADING):
            return self._parse_heading()

        # Rule
        if self._check(TokenType.RULE):
            return self._parse_rule()

        # Table
        if self._check(TokenType.TABLE_ROW_START):
            return self._parse_table()

        # Pre-formatted block
        if self._check(TokenType.PRE_BLOCK):
            return self._parse_pre_block()

        # Alignment block
        if self._check(TokenType.ALIGN_RIGHT) or self._check(TokenType.ALIGN_LEFT):
            return self._parse_align_block()

        # Solution/Hint
        if self._check(TokenType.SOLUTION_START):
            return self._parse_solution()

        if self._check(TokenType.HINT_START):
            return self._parse_hint()

        # Paragraph (default)
        return self._parse_paragraph()

    def _parse_math_block(self) -> MathBlock:
        """Parse display math block [```...```]."""
        self._advance()  # consume [```

        content = ""
        while not self._check(TokenType.MATH_BLOCK_END) and not self._is_at_end():
            content += self._advance().value

        self._advance()  # consume ```]

        return MathBlock(content=content.strip())

    def _parse_list(self) -> List:
        """Parse a list (ordered or unordered)."""
        items: list[ListItem] = []
        ordered = self._check(TokenType.ORDERED_ITEM)

        while not self._is_at_end():
            # Check if we have a list item
            if not (self._check(TokenType.LIST_ITEM) or self._check(TokenType.ORDERED_ITEM)):
                break

            token = self._advance()

            # Extract content from [* content or [1. content (no closing bracket)
            content_text = token.value
            if token.type == TokenType.LIST_ITEM:
                # Remove [* prefix and strip whitespace
                content_text = content_text[2:].strip()
            else:  # ORDERED_ITEM
                # Remove [N. prefix and strip whitespace
                content_text = content_text.split(".", 1)[1].strip()

            # Parse inline content
            item_tokens = PGMLTokenizer(content_text).tokenize()
            item_parser = PGMLParser(item_tokens)
            inline_content = item_parser._parse_inline_elements()

            items.append(ListItem(content=inline_content))

            # Skip newline after list item (if present)
            self._match(TokenType.NEWLINE)

        return List(items=items, ordered=ordered)

    def _parse_table(self) -> Table:
        """Parse a table with pipe-delimited cells: | cell1 | cell2 |"""
        rows: list[TableRow] = []

        # Parse all consecutive table rows
        while self._check(TokenType.TABLE_ROW_START) and not self._is_at_end():
            rows.append(self._parse_table_row())
            # Skip newline after row
            self._match(TokenType.NEWLINE)

        return Table(rows=rows)

    def _parse_table_row(self) -> TableRow:
        """Parse a single table row."""
        self._advance()  # consume TABLE_ROW_START (|)

        cells: list[list[PGMLNode]] = []
        current_cell: list[PGMLNode] = []

        # Parse cells until end of row
        while not self._check(TokenType.TABLE_ROW_END) and not self._is_at_end():
            if self._check(TokenType.TABLE_CELL_SEP):
                # End of current cell, start new one
                cells.append(current_cell)
                current_cell = []
                self._advance()  # consume separator
            elif self._check(TokenType.NEWLINE):
                # Unexpected newline in table
                break
            else:
                # Add inline content to current cell
                if self._check(TokenType.TEXT):
                    current_cell.append(
                        Text(content=self._advance().value.strip()))
                elif self._check(TokenType.VAR_START):
                    current_cell.append(self._parse_variable())
                elif self._check(TokenType.CODE_START):
                    current_cell.append(self._parse_code())
                elif self._check(TokenType.MATH_INLINE_START):
                    current_cell.append(self._parse_math_inline())
                else:
                    # Unknown token, skip
                    self._advance()

        # Add last cell
        if current_cell or cells:  # Don't add empty cell if no cells parsed
            cells.append(current_cell)

        # Consume TABLE_ROW_END if present
        if self._check(TokenType.TABLE_ROW_END):
            self._advance()

        return TableRow(cells=cells)

    def _parse_heading(self) -> Heading:
        """Parse heading: # Heading, ## Subheading, etc."""
        token = self._advance()  # consume heading token

        # Count # characters to determine level
        level = 0
        for char in token.value:
            if char == '#':
                level += 1
            else:
                break
        level = min(level, 6)  # Cap at 6

        # Extract heading text (remove leading # and whitespace)
        heading_text = token.value.lstrip('#').strip()

        # Parse heading content as inline elements
        content = [Text(content=heading_text)]

        return Heading(level=level, content=content)

    def _parse_rule(self) -> Rule:
        """Parse horizontal rule: --- or ==="""
        self._advance()  # consume rule token
        return Rule()

    def _parse_solution(self) -> Solution:
        """Parse solution section: BEGIN_PGML_SOLUTION ... END_PGML_SOLUTION"""
        self._advance()  # consume SOLUTION_START

        # Parse blocks until SOLUTION_END
        blocks: list[PGMLNode] = []
        while not self._check(TokenType.SOLUTION_END) and not self._is_at_end():
            # Skip blank lines
            while self._match(TokenType.BLANK_LINE, TokenType.NEWLINE):
                pass

            if self._check(TokenType.SOLUTION_END) or self._is_at_end():
                break

            block = self._parse_block()
            if block:
                blocks.append(block)

        # Consume SOLUTION_END
        if self._check(TokenType.SOLUTION_END):
            self._advance()

        return Solution(content=blocks)

    def _parse_hint(self) -> Hint:
        """Parse hint section: BEGIN_PGML_HINT ... END_PGML_HINT"""
        self._advance()  # consume HINT_START

        # Parse blocks until HINT_END
        blocks: list[PGMLNode] = []
        while not self._check(TokenType.HINT_END) and not self._is_at_end():
            # Skip blank lines
            while self._match(TokenType.BLANK_LINE, TokenType.NEWLINE):
                pass

            if self._check(TokenType.HINT_END) or self._is_at_end():
                break

            block = self._parse_block()
            if block:
                blocks.append(block)

        # Consume HINT_END
        if self._check(TokenType.HINT_END):
            self._advance()

        return Hint(content=blocks)

    def _parse_paragraph(self) -> Paragraph:
        """Parse a paragraph (inline content until blank line)."""
        content = self._parse_inline_elements()
        return Paragraph(content=content)

    def _parse_inline_elements(self) -> list[PGMLNode]:
        """Parse inline elements until block boundary."""
        elements: list[PGMLNode] = []

        while not self._is_at_block_boundary() and not self._is_at_end():
            # Variable interpolation
            if self._check(TokenType.VAR_START):
                elements.append(self._parse_variable())

            # Answer blank
            elif self._check(TokenType.ANSWER_BLANK):
                elements.append(self._parse_answer_blank())

            # Code execution
            elif self._check(TokenType.CODE_START):
                elements.append(self._parse_code())

            # Inline math
            elif self._check(TokenType.MATH_INLINE_START):
                elements.append(self._parse_math_inline())

            # Bold
            elif self._check(TokenType.BOLD_START):
                elements.append(self._parse_bold())

            # Italic
            elif self._check(TokenType.ITALIC_START):
                elements.append(self._parse_italic())

            # Plain text
            elif self._check(TokenType.TEXT):
                elements.append(Text(content=self._advance().value))

            # Single newline (space in output)
            elif self._check(TokenType.NEWLINE):
                self._advance()
                elements.append(Text(content=" "))

            else:
                # Skip unknown token
                self._advance()

        return elements

    def _parse_variable(self) -> Variable:
        """Parse variable interpolation [$var]."""
        self._advance()  # consume [$

        var_name = ""
        if self._check(TokenType.TEXT):
            var_name = self._advance().value

        self._match(TokenType.VAR_END)  # consume ]

        return Variable(name=var_name)

    def _parse_answer_blank(self) -> AnswerBlank:
        """Parse answer blank [_____] with optional {evaluator}."""
        token = self._advance()

        # Count underscores for width
        width = token.value.count("_")

        # Check for evaluator syntax: {$ans} or {evaluator_code}
        evaluator_code = ""
        if self._check(TokenType.TEXT):
            next_text = self.current_token.value
            if next_text.strip().startswith("{"):
                # Find matching closing brace
                brace_depth = 0
                eval_text = ""
                found_opening = False

                for char in next_text:
                    if char == "{":
                        brace_depth += 1
                        found_opening = True
                    elif char == "}":
                        brace_depth -= 1
                        if brace_depth == 0 and found_opening:
                            # Consume this text token
                            self._advance()
                            break

                    if found_opening and char not in "{}":
                        eval_text += char

                evaluator_code = eval_text.strip()

        return AnswerBlank(width=width, evaluator_code=evaluator_code)

    def _parse_code(self) -> Code:
        """Parse code execution [@code@] or [@code@]*."""
        self._advance()  # consume [@

        code = ""
        if self._check(TokenType.TEXT):
            code = self._advance().value

        # Check for @] or @]*
        # Get the token before matching to check for *
        if self._check(TokenType.CODE_END):
            end_token = self.current_token
            self._advance()  # consume the end token
            display_result = end_token.value.endswith("*")
        else:
            display_result = False

        return Code(code=code.strip(), display_result=display_result)

    def _parse_math_inline(self) -> MathInline:
        """Parse inline math [``...``]."""
        self._advance()  # consume [``

        content = ""
        if self._check(TokenType.TEXT):
            content = self._advance().value

        self._match(TokenType.MATH_INLINE_END)  # consume ``]

        return MathInline(content=content.strip())

    def _parse_bold(self) -> Bold:
        """Parse bold text *...* or **...**."""
        marker = self._advance().value  # * or **
        content: list[PGMLNode] = []

        # Parse content until matching end marker
        while not self._check(TokenType.BOLD_END) and not self._is_at_end():
            if self._check(TokenType.TEXT):
                content.append(Text(content=self._advance().value))
            elif self._check(TokenType.NEWLINE):
                self._advance()
                content.append(Text(content=" "))
            else:
                break

        self._match(TokenType.BOLD_END)

        return Bold(content=content)

    def _parse_italic(self) -> Italic:
        """Parse italic text _..._ or __...__."""
        marker = self._advance().value  # _ or __
        content: list[PGMLNode] = []

        # Parse content until matching end marker
        while not self._check(TokenType.ITALIC_END) and not self._is_at_end():
            if self._check(TokenType.TEXT):
                content.append(Text(content=self._advance().value))
            elif self._check(TokenType.NEWLINE):
                self._advance()
                content.append(Text(content=" "))
            else:
                break

        self._match(TokenType.ITALIC_END)

        return Italic(content=content)

    def _is_at_block_boundary(self) -> bool:
        """Check if at a block boundary."""
        return self._check(
            TokenType.BLANK_LINE,
            TokenType.MATH_BLOCK_START,
            TokenType.LIST_ITEM,
            TokenType.ORDERED_ITEM,
            TokenType.EOF,
        )

    def _check(self, *token_types: TokenType) -> bool:
        """Check if current token matches any of the given types."""
        if self._is_at_end():
            return False
        return self.current_token.type in token_types

    def _match(self, *token_types: TokenType) -> bool:
        """Try to match and consume any of the given token types."""
        if self._check(*token_types):
            self._advance()
            return True
        return False

    def _advance(self) -> Token:
        """Consume and return current token."""
        token = self.current_token
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        return token

    def _is_at_end(self) -> bool:
        """Check if at end of token stream."""
        return self.current_token is None or self.current_token.type == TokenType.EOF

    def _parse_align_block(self) -> AlignBlock:
        """
        Parse alignment block:
        >> right-aligned
        << left-aligned  
        >> centered <<
        
        Returns:
            AlignBlock with alignment type and content
        """
        first_token = self._advance()
        
        # Determine alignment
        if first_token.type == TokenType.ALIGN_RIGHT:
            # Check if this is center (>> ... <<)
            content_nodes = []
            
            while not self._is_at_end():
                if self._check(TokenType.ALIGN_LEFT):
                    # Found closing <<, this is center
                    self._advance()
                    return AlignBlock(alignment="center", content=content_nodes)
                
                if self._check(TokenType.NEWLINE) or self._check(TokenType.BLANK_LINE):
                    # End of line, this is right-align
                    break
                
                # Parse inline content
                if self._check(TokenType.TEXT):
                    content_nodes.append(Text(content=self._advance().value))
                else:
                    self._advance()  # Skip other tokens
            
            return AlignBlock(alignment="right", content=content_nodes)
        
        else:  # ALIGN_LEFT
            # Left-aligned content
            content_nodes = []
            
            while not self._is_at_end():
                if self._check(TokenType.NEWLINE) or self._check(TokenType.BLANK_LINE):
                    break
                
                if self._check(TokenType.TEXT):
                    content_nodes.append(Text(content=self._advance().value))
                else:
                    self._advance()
            
            return AlignBlock(alignment="left", content=content_nodes)
