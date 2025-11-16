"""
PGML (PG Markup Language) Parser.

Converts PGML markdown-like syntax to HTML with answer blanks.

PGML Syntax Reference:
- Variable interpolation: [$var]
- Answer blanks: [_]{$evaluator} or [__]{$evaluator}{width}
- Code evaluation: [@ code @] or [@* code @]*
- Inline math: [`formula`] or [` formula `]
- Display math: [`` formula ``]
- Bold: *text* or **text**
- Italic: _text_ (when not answer blank)
- Lists: + item, - item, * item
- Headings: #, ##, ###
- Horizontal rule: ---
- Line break: two spaces at end of line
- Paragraph: blank line

Reference: macros/core/PGML.pl
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PGMLNode:
    """Base class for PGML AST nodes."""

    type: str


@dataclass
class TextNode(PGMLNode):
    """Plain text node."""

    text: str
    type: str = field(default="text", init=False)


@dataclass
class VariableNode(PGMLNode):
    """Variable interpolation: [$var]"""

    var_name: str
    type: str = field(default="variable", init=False)


@dataclass
class AnswerBlankNode(PGMLNode):
    """Answer blank: [_]{$ans} or [___]{$ans}{20}"""

    evaluator_expr: str
    width: int = 20
    type: str = field(default="answer_blank", init=False)


@dataclass
class CodeNode(PGMLNode):
    """Code evaluation: [@ code @] or [@* code @]*"""

    code: str
    has_star: bool = False
    type: str = field(default="code", init=False)


@dataclass
class MathNode(PGMLNode):
    """Math expression: [`x^2`] or [``x^2``]"""

    math: str
    display: bool = False
    type: str = field(default="math", init=False)


@dataclass
class BoldNode(PGMLNode):
    """Bold text: *text* or **text**"""

    children: list[PGMLNode] = field(default_factory=list)
    type: str = field(default="bold", init=False)


@dataclass
class ItalicNode(PGMLNode):
    """Italic text: _text_"""

    children: list[PGMLNode] = field(default_factory=list)
    type: str = field(default="italic", init=False)


@dataclass
class ListNode(PGMLNode):
    """List with items."""

    items: list[list[PGMLNode]] = field(default_factory=list)
    ordered: bool = False
    type: str = field(default="list", init=False)


@dataclass
class HeadingNode(PGMLNode):
    """Heading: # text"""

    level: int = 1
    children: list[PGMLNode] = field(default_factory=list)
    type: str = field(default="heading", init=False)


@dataclass
class RuleNode(PGMLNode):
    """Horizontal rule: ---"""

    type: str = field(default="rule", init=False)


@dataclass
class ParNode(PGMLNode):
    """Paragraph break."""

    type: str = field(default="par", init=False)


@dataclass
class LineBreakNode(PGMLNode):
    """Line break: two spaces at end of line."""

    type: str = field(default="break", init=False)


@dataclass
class PGMLDocument:
    """PGML document with nodes."""

    nodes: list[PGMLNode] = field(default_factory=list)


class PGMLParser:
    """
    Parser for PGML markup language.

    Converts PGML text to an AST that can be rendered to HTML.
    """

    def __init__(self):
        """Initialize parser."""
        # Patterns for PGML constructs
        self.patterns = {
            # Display math: [`` math ``] - must come before inline math
            "display_math": re.compile(r"\[``([^`]*?)``\]", re.DOTALL),
            # Inline math: [` math `] - non-greedy, no nested backticks
            "inline_math": re.compile(r"\[`([^`]*?)`\]", re.DOTALL),
            # Answer blank: [_]{$ans} or [___]{$ans}{20}
            "answer_blank": re.compile(r"\[(_+)\]\{([^}]+)\}(?:\{(\d+)\})?"),
            # Variable: [$var] or [$var->method()]
            "variable": re.compile(r"\[\$([^\]]+)\]"),
            # Code: [@ code @] or [@* code @]*
            "code": re.compile(r"\[@(\*?)(.*?)@\](\*?)", re.DOTALL),
            # Bold: **text** or *text*
            "bold": re.compile(r"\*\*(.+?)\*\*|\*([^\*\s][^\*]*?[^\*\s])\*"),
            # Italic: _text_
            "italic": re.compile(r"_([^_\s][^_]*?[^_\s])_"),
        }

    def parse(self, text: str, context: dict[str, Any] | None = None) -> PGMLDocument:
        """
        Parse PGML text into document AST.

        Args:
            text: PGML markup text
            context: Optional context dict with variables

        Returns:
            PGMLDocument with parsed nodes
        """
        self.context = context or {}
        doc = PGMLDocument()

        # Split into lines for block-level parsing
        lines = text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check for list items
            list_match = re.match(r"^([+\-*]) (.+)$", line)
            if list_match:
                # Parse list
                list_node, i = self._parse_list(lines, i)
                doc.nodes.append(list_node)
                continue

            # Check for headings
            heading_match = re.match(r"^(#{1,6}) (.+)$", line)
            if heading_match:
                level = len(heading_match.group(1))
                content = heading_match.group(2)
                heading = HeadingNode(
                    level=level, children=self._parse_inline(content)
                )
                doc.nodes.append(heading)
                i += 1
                continue

            # Check for horizontal rule
            if re.match(r"^---+$", line.strip()):
                doc.nodes.append(RuleNode())
                i += 1
                continue

            # Check for blank line (paragraph break)
            if not line.strip():
                doc.nodes.append(ParNode())
                i += 1
                continue

            # Regular paragraph - parse inline content
            inline_nodes = self._parse_inline(line)
            doc.nodes.extend(inline_nodes)

            # Check for line break (two spaces at end)
            if i < len(lines) - 1 and line.endswith("  "):
                doc.nodes.append(LineBreakNode())

            i += 1

        return doc

    def _parse_list(
        self, lines: list[str], start: int
    ) -> tuple[ListNode, int]:
        """Parse list starting at given line."""
        list_node = ListNode()
        i = start

        while i < len(lines):
            line = lines[i]
            match = re.match(r"^([+\-*]) (.+)$", line)
            if not match:
                break

            # Parse list item content
            content = match.group(2)
            item_nodes = self._parse_inline(content)
            list_node.items.append(item_nodes)

            i += 1

        return list_node, i

    def _parse_inline(self, text: str) -> list[PGMLNode]:
        """
        Parse inline PGML elements (variables, math, bold, etc.).

        Args:
            text: Text to parse

        Returns:
            List of parsed nodes
        """
        nodes: list[PGMLNode] = []
        pos = 0

        # Find all matches for all patterns
        matches: list[tuple[int, int, str, re.Match]] = []

        for name, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                matches.append((match.start(), match.end(), name, match))

        # Sort by position
        matches.sort(key=lambda x: x[0])

        # Process matches in order
        for start, end, name, match in matches:
            # Skip overlapping matches
            if start < pos:
                continue

            # Add text before match
            if pos < start:
                text_content = text[pos:start]
                if text_content:
                    nodes.append(TextNode(text=text_content))

            # Create node based on type
            if name == "answer_blank":
                evaluator = match.group(2)
                width_str = match.group(3)
                width = int(width_str) if width_str else 20
                nodes.append(
                    AnswerBlankNode(evaluator_expr=evaluator, width=width)
                )

            elif name == "variable":
                var_name = match.group(1)
                nodes.append(VariableNode(var_name=var_name))

            elif name == "code":
                star_before = match.group(1)
                code = match.group(2).strip()
                star_after = match.group(3)
                has_star = bool(star_before or star_after)
                nodes.append(CodeNode(code=code, has_star=has_star))

            elif name == "display_math":
                math = match.group(1).strip()
                nodes.append(MathNode(math=math, display=True))

            elif name == "inline_math":
                math = match.group(1).strip()
                nodes.append(MathNode(math=math, display=False))

            elif name == "bold":
                content = match.group(1) or match.group(2)
                # Recursively parse content
                children = self._parse_inline(content)
                nodes.append(BoldNode(children=children))

            elif name == "italic":
                content = match.group(1)
                # Recursively parse content
                children = self._parse_inline(content)
                nodes.append(ItalicNode(children=children))

            pos = end

        # Add remaining text
        if pos < len(text):
            remaining = text[pos:]
            if remaining:
                nodes.append(TextNode(text=remaining))

        return nodes


class PGMLRenderer:
    """
    Renders PGML AST to HTML.

    Handles answer blank insertion, variable substitution, and formatting.
    """

    def __init__(self, context: dict[str, Any] | None = None):
        """
        Initialize renderer.

        Args:
            context: Variable context for interpolation
        """
        self.context = context or {}
        self.answer_counter = 0

    def render(self, doc: PGMLDocument) -> str:
        """
        Render PGML document to HTML.

        Args:
            doc: Parsed PGML document

        Returns:
            HTML string
        """
        parts: list[str] = []

        for node in doc.nodes:
            html = self._render_node(node)
            parts.append(html)

        return "".join(parts)

    def _render_node(self, node: PGMLNode) -> str:
        """Render a single node to HTML."""
        if isinstance(node, TextNode):
            return self._escape_html(node.text)

        elif isinstance(node, VariableNode):
            # Get variable value from context
            value = self._get_variable(node.var_name)
            return str(value)

        elif isinstance(node, AnswerBlankNode):
            self.answer_counter += 1
            return f'<input type="text" name="AnSwEr{self.answer_counter:04d}" size="{node.width}" />'

        elif isinstance(node, CodeNode):
            # Code nodes should be evaluated during preprocessing
            return f'<code>{self._escape_html(node.code)}</code>'

        elif isinstance(node, MathNode):
            if node.display:
                return f'\\[{node.math}\\]'
            else:
                return f'\\({node.math}\\)'

        elif isinstance(node, BoldNode):
            content = "".join(self._render_node(child)
                              for child in node.children)
            return f"<b>{content}</b>"

        elif isinstance(node, ItalicNode):
            content = "".join(self._render_node(child)
                              for child in node.children)
            return f"<i>{content}</i>"

        elif isinstance(node, ListNode):
            tag = "ol" if node.ordered else "ul"
            items = []
            for item_nodes in node.items:
                item_html = "".join(self._render_node(n) for n in item_nodes)
                items.append(f"<li>{item_html}</li>")
            return f'<{tag}>{"".join(items)}</{tag}>'

        elif isinstance(node, HeadingNode):
            content = "".join(self._render_node(child)
                              for child in node.children)
            return f"<h{node.level}>{content}</h{node.level}>"

        elif isinstance(node, RuleNode):
            return "<hr />"

        elif isinstance(node, ParNode):
            return "<p></p>"

        elif isinstance(node, LineBreakNode):
            return "<br />"

        else:
            return ""

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def _get_variable(self, var_name: str) -> Any:
        """Get variable value from context."""
        # Handle method calls like $var->method()
        if "->" in var_name:
            var_name = var_name.split("->")[0]

        return self.context.get(var_name, f"[${var_name}]")
