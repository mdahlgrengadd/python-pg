"""
Block extraction for PG special blocks.

This module handles extraction of BEGIN_TEXT, BEGIN_PGML, BEGIN_SOLUTION,
BEGIN_HINT, and other special blocks from PG source code.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BlockExtractionResult:
    """Result of block extraction attempt."""

    found: bool
    """Whether a block was found"""

    block_type: Optional[str]
    """Type of block (TEXT, PGML, SOLUTION, etc.)"""

    content: str
    """Content of the block"""

    lines_consumed: int
    """Number of lines consumed (including BEGIN/END markers)"""


class PGBlockExtractor:
    """
    Extracts special PG blocks from source code.

    This class handles detection and extraction of special blocks like
    BEGIN_TEXT...END_TEXT, BEGIN_PGML...END_PGML, etc. It also handles
    method-style blocks like $obj->BEGIN_TIKZ...END_TIKZ.
    """

    # Block patterns: (begin_pattern, end_pattern)
    BLOCK_PATTERNS: Dict[str, Tuple[str, str]] = {
        "TEXT": (r"BEGIN_TEXT\s*$", r"^END_TEXT"),
        "PGML": (r"BEGIN_PGML\s*$", r"^END_PGML"),
        "SOLUTION": (r"BEGIN_SOLUTION\s*$", r"^END_SOLUTION"),
        "HINT": (r"BEGIN_HINT\s*$", r"^END_HINT"),
        "PGML_SOLUTION": (r"BEGIN_PGML_SOLUTION\s*$", r"^END_PGML_SOLUTION"),
        "PGML_HINT": (r"BEGIN_PGML_HINT\s*$", r"^END_PGML_HINT"),
        "TIKZ": (r"BEGIN_TIKZ\s*$", r"^END_TIKZ"),
    }

    def is_special_block_marker(self, line: str) -> bool:
        """
        Check if a line is a special block marker.

        Args:
            line: The line to check

        Returns:
            bool: True if the line starts a special block
        """
        stripped = line.strip()
        return any(
            re.match(pattern, stripped)
            for pattern, _ in self.BLOCK_PATTERNS.values()
        )

    def extract_block(
        self,
        lines: List[str],
        start_index: int
    ) -> Optional[BlockExtractionResult]:
        """
        Extract a special block starting at the given line index.

        Args:
            lines: List of source lines
            start_index: Index of the line to check for block start

        Returns:
            BlockExtractionResult if a block is found, None otherwise
        """
        if start_index >= len(lines):
            return None

        current_line = lines[start_index]

        # Check for standard blocks (BEGIN_TEXT, BEGIN_PGML, etc.)
        for block_type, (begin_pattern, end_pattern) in self.BLOCK_PATTERNS.items():
            # Skip method-call style blocks (handled separately)
            if re.search(begin_pattern, current_line) and not re.search(
                r'\$\w+(?:\[[^\]]*\])?\s*->\s*(BEGIN_TIKZ|BEGIN_LATEX_IMAGE)',
                current_line
            ):
                block_content_lines: List[str] = []
                i = start_index + 1

                # Collect lines until we find the end marker
                while i < len(lines) and not re.match(end_pattern, lines[i]):
                    block_content_lines.append(lines[i])
                    i += 1

                block_content = "\n".join(block_content_lines)
                lines_consumed = i - start_index + 1  # Include END marker

                return BlockExtractionResult(
                    found=True,
                    block_type=block_type,
                    content=block_content,
                    lines_consumed=lines_consumed
                )

        return None

    def extract_method_block(
        self,
        lines: List[str],
        start_index: int
    ) -> Optional[Tuple[str, str, str, int]]:
        """
        Extract method-style blocks like $obj->BEGIN_TIKZ...END_TIKZ.

        Args:
            lines: List of source lines
            start_index: Index of the line to check

        Returns:
            Tuple of (obj_var, method_name, content, lines_consumed) if found,
            None otherwise
        """
        if start_index >= len(lines):
            return None

        current_line = lines[start_index]

        # Check for method-call-style blocks: $obj->BEGIN_TIKZ or $obj->BEGIN_LATEX_IMAGE
        # Also handle array indexing like $graph[$i]->BEGIN_TIKZ
        tikz_method_match = re.search(
            r'(\$\w+(?:\[[^\]]*\])*)\s*->\s*BEGIN_TIKZ', current_line
        )
        latex_method_match = re.search(
            r'(\$\w+(?:\[[^\]]*\])*)\s*->\s*BEGIN_LATEX_IMAGE', current_line
        )

        if (tikz_method_match or latex_method_match) and current_line.strip().endswith(
            ('BEGIN_TIKZ', 'BEGIN_LATEX_IMAGE')
        ):
            obj_var = (tikz_method_match or latex_method_match).group(1)
            end_marker = "END_TIKZ" if tikz_method_match else "END_LATEX_IMAGE"
            method_name = "BEGIN_TIKZ" if tikz_method_match else "BEGIN_LATEX_IMAGE"

            # Collect content lines until we hit the end marker
            content_lines: List[str] = []
            i = start_index + 1
            while i < len(lines):
                if re.match(rf'^\s*{end_marker}\s*$', lines[i]):
                    break
                content_lines.append(lines[i])
                i += 1

            # Join content
            content = '\n'.join(content_lines)
            lines_consumed = i - start_index + 1  # Include END marker

            return (obj_var, method_name, content, lines_consumed)

        return None

    def transform_pgml_evaluators(self, pgml_content: str) -> str:
        """
        Transform PGML evaluator syntax [@ ... @] to Python.

        Converts Perl-like code inside PGML evaluators to Python syntax.
        Handles variable dereferencing and basic operators.

        Args:
            pgml_content: Raw PGML content with [@ ... @] evaluators

        Returns:
            Transformed PGML content with converted evaluators
        """
        def replace_evaluator(match):
            perl_expr = match.group(1).strip()

            # Handle simple variable dereferencing: $var -> var
            # But preserve ${...} syntax for complex expressions
            if re.match(r'^\$\w+$', perl_expr):
                return f'[@ {perl_expr[1:]} @]'

            # Handle array/hash access: $var[0] -> var[0], $hash{key} -> hash["key"]
            # Replace ${...} with {...} for hash access
            perl_expr = re.sub(r'\$(\w+)\[(\d+)\]', r'\1[\2]', perl_expr)
            perl_expr = re.sub(r'\$\{([^}]+)\}', r'{\1}', perl_expr)

            # Replace -> with . for method calls
            perl_expr = perl_expr.replace('->', '.')

            # Remove $ sigils from variables (but not from special vars like $_)
            perl_expr = re.sub(r'\$(\w+)', r'\1', perl_expr)

            return f'[@ {perl_expr} @]'

        # Replace all [@ ... @] evaluators
        return re.sub(r'\[@\s*(.*?)\s*@\]', replace_evaluator, pgml_content, flags=re.DOTALL)

    def escape_triple_quotes(self, text: str) -> str:
        """
        Escape triple quotes in text for use in Python triple-quoted strings.

        Args:
            text: Text that may contain triple quotes

        Returns:
            Text with escaped triple quotes
        """
        # Escape ''' by breaking it into '' + '
        return text.replace("'''", r"''\'''")
