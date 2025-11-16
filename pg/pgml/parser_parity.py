"""
PGML Parser Parity Extensions - New parsing methods for 100% parity.

This file contains parsing methods for the new PGML features:
- Headings
- Tables  
- Rules
- Alignment blocks
- Pre-formatted blocks
- Solutions/Hints

These will be integrated into the main parser.py file.

Reference: macros/core/PGML.pl (lines 800-1200)
"""

from .parser import PGMLNode, Heading, Table, TableRow, Rule, PreBlock, AlignBlock, Solution, Hint, Text
from .tokenizer import TokenType, Token


# Methods to add to PGMLParser class

def _parse_heading(self) -> Heading:
    """
    Parse heading: # Heading, ## Subheading, etc.
    
    Returns:
        Heading node with level and content
    """
    token = self._advance()  # consume HEADING token
    
    # Extract level (count # characters)
    heading_text = token.value
    level = 0
    for ch in heading_text:
        if ch == "#":
            level += 1
        else:
            break
    
    # Extract content (after # and optional space)
    content_text = heading_text[level:].strip()
    
    # Parse inline content
    from .tokenizer import PGMLTokenizer
    inline_tokens = PGMLTokenizer(content_text).tokenize()
    inline_parser = type(self)(inline_tokens)
    content = inline_parser._parse_inline_elements()
    
    return Heading(level=level, content=content)


def _parse_rule(self) -> Rule:
    """
    Parse horizontal rule: --- or ===.
    
    Returns:
        Rule node with style indicator
    """
    token = self._advance()  # consume RULE token
    
    # Determine style from first character
    style = token.value[0]  # "-" or "="
    
    return Rule(style=style)


def _parse_table(self) -> Table:
    """
    Parse table with rows and cells.
    
    Format:
    | cell1 | cell2 | cell3 |
    | data1 | data2 | data3 |
    
    Returns:
        Table node with rows containing cells
    """
    rows = []
    
    while not self._is_at_end() and self._check(TokenType.TABLE_ROW_START):
        row = self._parse_table_row()
        rows.append(row)
        
        # Skip newlines between rows
        while self._check(TokenType.NEWLINE) or self._check(TokenType.BLANK_LINE):
            self._advance()
    
    return Table(rows=rows)


def _parse_table_row(self) -> TableRow:
    """
    Parse a single table row: | cell1 | cell2 |
    
    Returns:
        TableRow with list of cells (each cell is list of inline nodes)
    """
    self._advance()  # consume TABLE_ROW_START
    
    cells = []
    current_cell_text = ""
    
    while not self._is_at_end():
        if self._check(TokenType.TABLE_CELL_SEP):
            # End of cell
            if current_cell_text.strip():
                # Parse cell content as inline elements
                from .tokenizer import PGMLTokenizer
                cell_tokens = PGMLTokenizer(current_cell_text.strip()).tokenize()
                cell_parser = type(self)(cell_tokens)
                cell_content = cell_parser._parse_inline_elements()
                cells.append(cell_content)
            else:
                cells.append([])  # Empty cell
            
            current_cell_text = ""
            self._advance()  # consume separator
            
        elif self._check(TokenType.TABLE_ROW_END):
            # End of row
            if current_cell_text.strip():
                from .tokenizer import PGMLTokenizer
                cell_tokens = PGMLTokenizer(current_cell_text.strip()).tokenize()
                cell_parser = type(self)(cell_tokens)
                cell_content = cell_parser._parse_inline_elements()
                cells.append(cell_content)
            
            self._advance()  # consume row end
            break
            
        elif self._check(TokenType.TEXT):
            current_cell_text += self._advance().value
            
        else:
            break
    
    return TableRow(cells=cells)


def _parse_pre_block(self) -> PreBlock:
    """
    Parse pre-formatted block: :   content
    
    Returns:
        PreBlock node with raw content
    """
    token = self._advance()  # consume PRE_BLOCK token
    
    # Extract content (after :   prefix)
    content = token.value[4:] if len(token.value) > 4 else ""
    
    return PreBlock(content=content)


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


def _parse_solution(self) -> Solution:
    """
    Parse solution section: BEGIN_PGML_SOLUTION ... END_PGML_SOLUTION.
    
    Returns:
        Solution node with blocks
    """
    self._advance()  # consume SOLUTION_START
    
    blocks = []
    
    while not self._is_at_end() and not self._check(TokenType.SOLUTION_END):
        # Skip blank lines
        while self._match(TokenType.BLANK_LINE, TokenType.NEWLINE):
            pass
        
        if self._is_at_end() or self._check(TokenType.SOLUTION_END):
            break
        
        # Parse block
        block = self._parse_block()
        if block:
            blocks.append(block)
    
    if self._check(TokenType.SOLUTION_END):
        self._advance()  # consume SOLUTION_END
    
    return Solution(content=blocks)


def _parse_hint(self) -> Hint:
    """
    Parse hint section: BEGIN_PGML_HINT ... END_PGML_HINT.
    
    Returns:
        Hint node with blocks
    """
    self._advance()  # consume HINT_START
    
    blocks = []
    
    while not self._is_at_end() and not self._check(TokenType.HINT_END):
        # Skip blank lines
        while self._match(TokenType.BLANK_LINE, TokenType.NEWLINE):
            pass
        
        if self._is_at_end() or self._check(TokenType.HINT_END):
            break
        
        # Parse block
        block = self._parse_block()
        if block:
            blocks.append(block)
    
    if self._check(TokenType.HINT_END):
        self._advance()  # consume HINT_END
    
    return Hint(content=blocks)


# These methods should be added to the PGMLParser class in parser.py
# They extend the existing parser with full PGML.pl parity features

