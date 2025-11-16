"""
PGML Renderers - Convert PGML AST to HTML or TeX.

Implements the Visitor pattern to traverse the AST and generate output.

Reference: PGML.pl rendering logic (lines 1200-1800)
"""

from abc import ABC, abstractmethod
from typing import Any

from .parser import (
    AnswerBlank,
    Bold,
    Code,
    Document,
    Italic,
    List,
    ListItem,
    MathBlock,
    MathInline,
    Paragraph,
    PGMLNode,
    Text,
    Variable,
)


class Renderer(ABC):
    """Base class for PGML renderers."""

    @abstractmethod
    def render(self, node: PGMLNode) -> str:
        """Render a PGML node to output format."""
        pass

    @abstractmethod
    def visit_document(self, node: Document) -> str:
        pass

    @abstractmethod
    def visit_paragraph(self, node: Paragraph) -> str:
        pass

    @abstractmethod
    def visit_math_block(self, node: MathBlock) -> str:
        pass

    @abstractmethod
    def visit_list(self, node: List) -> str:
        pass

    @abstractmethod
    def visit_list_item(self, node: ListItem) -> str:
        pass

    @abstractmethod
    def visit_text(self, node: Text) -> str:
        pass

    @abstractmethod
    def visit_variable(self, node: Variable) -> str:
        pass

    @abstractmethod
    def visit_answer_blank(self, node: AnswerBlank) -> str:
        pass

    @abstractmethod
    def visit_code(self, node: Code) -> str:
        pass

    @abstractmethod
    def visit_math_inline(self, node: MathInline) -> str:
        pass

    @abstractmethod
    def visit_bold(self, node: Bold) -> str:
        pass

    @abstractmethod
    def visit_italic(self, node: Italic) -> str:
        pass

    # NEW FOR PARITY: Abstract methods for new node types

    @abstractmethod
    def visit_heading(self, node: Any) -> str:
        pass

    @abstractmethod
    def visit_rule(self, node: Any) -> str:
        pass

    @abstractmethod
    def visit_table(self, node: Any) -> str:
        pass

    @abstractmethod
    def visit_table_row(self, node: Any) -> str:
        pass

    @abstractmethod
    def visit_align_block(self, node: Any) -> str:
        pass

    @abstractmethod
    def visit_pre_block(self, node: Any) -> str:
        pass

    @abstractmethod
    def visit_solution(self, node: Any) -> str:
        pass

    @abstractmethod
    def visit_hint(self, node: Any) -> str:
        pass


class HTMLRenderer(Renderer):
    """
    Render PGML to HTML.

    Uses KaTeX-compatible syntax for math rendering.
    """

    def __init__(self, context: dict[str, Any] | None = None, answer_counter: int = 1, code_executor: Any | None = None):
        """
        Initialize HTML renderer.

        Args:
            context: Variable bindings for interpolation
            answer_counter: Starting number for answer blanks
            code_executor: Optional code executor with eval(code) method for [@...@] blocks
        """
        self.context = context or {}
        self.answer_counter = answer_counter
        self.code_executor = code_executor

    def render(self, node: PGMLNode) -> str:
        """Render a PGML node to HTML."""
        return node.accept(self)

    def visit_document(self, node: Document) -> str:
        blocks_html = "\n".join(block.accept(self) for block in node.blocks)
        return f'<div class="pgml-document">\n{blocks_html}\n</div>'

    def visit_paragraph(self, node: Paragraph) -> str:
        content_html = "".join(child.accept(self) for child in node.content)
        return f"<p>{content_html}</p>"

    def visit_math_block(self, node: MathBlock) -> str:
        # Use KaTeX display math delimiters
        escaped_math = self._escape_html(node.content)
        return f'<div class="math-block">\\[{escaped_math}\\]</div>'

    def visit_list(self, node: List) -> str:
        tag = "ol" if node.ordered else "ul"
        items_html = "\n".join(item.accept(self) for item in node.items)
        return f"<{tag}>\n{items_html}\n</{tag}>"

    def visit_list_item(self, node: ListItem) -> str:
        content_html = "".join(child.accept(self) for child in node.content)
        # Strip trailing whitespace from list items
        content_html = content_html.rstrip()
        return f"<li>{content_html}</li>"

    def visit_text(self, node: Text) -> str:
        return self._escape_html(node.content)

    def visit_variable(self, node: Variable) -> str:
        # Look up variable in context
        value = self.context.get(node.name, f"[${node.name}]")

        # Convert value to string
        if hasattr(value, "to_string"):
            value_str = value.to_string()
        else:
            value_str = str(value)

        return f'<span class="pgml-variable">{self._escape_html(value_str)}</span>'

    def visit_answer_blank(self, node: AnswerBlank) -> str:
        # Generate answer input field
        answer_name = node.name or f"AnSwEr{self.answer_counter:04d}"
        self.answer_counter += 1

        width_chars = node.width if node.width > 0 else 20
        # Approximate character width in pixels
        width_px = width_chars * 10

        # Register evaluator if we have code and context
        if node.evaluator_code and hasattr(self, '_register_answer'):
            try:
                # Remove Perl $ sigil if present
                eval_code = node.evaluator_code.lstrip('$')
                # Evaluate the evaluator code in the context
                evaluator = eval(eval_code, {}, self.context)
                self._register_answer(answer_name, evaluator)
            except Exception:
                pass  # Silently skip if evaluator can't be resolved

        return (
            f'<input type="text" '
            f'name="{answer_name}" '
            f'id="{answer_name}" '
            f'class="pgml-answer-blank" '
            f'size="{width_chars}" '
            f'style="width: {width_px}px;" '
            f'aria-label="Answer {self.answer_counter - 1}">'
        )

    def visit_code(self, node: Code) -> str:
        """Execute code block and interpolate result."""
        # If no executor provided, show placeholder
        if not self.code_executor:
            return f'<span class="pgml-code" data-code="{self._escape_html(node.code)}">[code result]</span>'

        try:
            # Execute code in the environment
            result = self.code_executor.eval(node.code)

            # If display_result is False ([@code@] without *), return empty string
            if not node.display_result:
                return ""

            # Convert result to string for display
            if result is None:
                return ""
            elif hasattr(result, 'to_tex'):  # MathValue object
                # Render as inline math
                return f'<span class="math-inline">\\({result.to_tex()}\\)</span>'
            elif hasattr(result, '__html__'):  # Has HTML representation
                return str(result.__html__())
            else:
                # Plain text result
                result_str = str(result)
                return self._escape_html(result_str)

        except Exception as e:
            # Show error in development, hide in production
            return f'<span class="pgml-code-error" title="{self._escape_html(str(e))}">[code error]</span>'

    def visit_math_inline(self, node: MathInline) -> str:
        # Use KaTeX inline math delimiters
        escaped_math = self._escape_html(node.content)
        return f'<span class="math-inline">\\({escaped_math}\\)</span>'

    def visit_bold(self, node: Bold) -> str:
        content_html = "".join(child.accept(self) for child in node.content)
        return f"<strong>{content_html}</strong>"

    def visit_italic(self, node: Italic) -> str:
        content_html = "".join(child.accept(self) for child in node.content)
        return f"<em>{content_html}</em>"

    # NEW FOR PARITY: HTML rendering for new node types

    def visit_heading(self, node: Any) -> str:
        """Render heading as <h1> through <h6>."""
        from .parser import Heading
        if not isinstance(node, Heading):
            return ""

        content_html = "".join(child.accept(self) for child in node.content)
        level = min(node.level, 6)  # Cap at h6
        return f"<h{level}>{content_html}</h{level}>"

    def visit_rule(self, node: Any) -> str:
        """Render horizontal rule."""
        return '<hr class="pgml-rule">'

    def visit_table(self, node: Any) -> str:
        """Render table with rows and cells."""
        from .parser import Table
        if not isinstance(node, Table):
            return ""

        rows_html = "\n".join(row.accept(self) for row in node.rows)
        return f'<table class="pgml-table">\n{rows_html}\n</table>'

    def visit_table_row(self, node: Any) -> str:
        """Render table row with cells."""
        from .parser import TableRow
        if not isinstance(node, TableRow):
            return ""

        cells_html = []
        for cell_content in node.cells:
            cell_html = "".join(child.accept(self) for child in cell_content)
            cells_html.append(f"<td>{cell_html}</td>")

        return f"<tr>{''.join(cells_html)}</tr>"

    def visit_align_block(self, node: Any) -> str:
        """Render alignment block."""
        from .parser import AlignBlock
        if not isinstance(node, AlignBlock):
            return ""

        content_html = "".join(child.accept(self) for child in node.content)
        align_class = f"text-{node.alignment}"
        return f'<div class="pgml-align {align_class}">{content_html}</div>'

    def visit_pre_block(self, node: Any) -> str:
        """Render pre-formatted block."""
        from .parser import PreBlock
        if not isinstance(node, PreBlock):
            return ""

        escaped = self._escape_html(node.content)
        return f'<pre class="pgml-pre">{escaped}</pre>'

    def visit_solution(self, node: Any) -> str:
        """Render solution section."""
        from .parser import Solution
        if not isinstance(node, Solution):
            return ""

        blocks_html = "\n".join(block.accept(self) for block in node.content)
        return f'<div class="pgml-solution">\n<h4>Solution:</h4>\n{blocks_html}\n</div>'

    def visit_hint(self, node: Any) -> str:
        """Render hint section."""
        from .parser import Hint
        if not isinstance(node, Hint):
            return ""

        blocks_html = "\n".join(block.accept(self) for block in node.content)
        return f'<div class="pgml-hint">\n<h4>Hint:</h4>\n{blocks_html}\n</div>'

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )


class TeXRenderer(Renderer):
    """
    Render PGML to LaTeX/TeX.

    Generates print-ready TeX output for hardcopy generation.
    """

    def __init__(self, context: dict[str, Any] | None = None):
        """
        Initialize TeX renderer.

        Args:
            context: Variable bindings for interpolation
        """
        self.context = context or {}

    def render(self, node: PGMLNode) -> str:
        """Render a PGML node to TeX."""
        return node.accept(self)

    def visit_document(self, node: Document) -> str:
        blocks_tex = "\n\n".join(block.accept(self) for block in node.blocks)
        return blocks_tex

    def visit_paragraph(self, node: Paragraph) -> str:
        content_tex = "".join(child.accept(self) for child in node.content)
        return content_tex

    def visit_math_block(self, node: MathBlock) -> str:
        # Use LaTeX display math
        return f"\\[\n{node.content}\n\\]"

    def visit_list(self, node: List) -> str:
        env = "enumerate" if node.ordered else "itemize"
        items_tex = "\n".join(item.accept(self) for item in node.items)
        return f"\\begin{{{env}}}\n{items_tex}\n\\end{{{env}}}"

    def visit_list_item(self, node: ListItem) -> str:
        content_tex = "".join(child.accept(self) for child in node.content)
        # Strip trailing whitespace from list items
        content_tex = content_tex.rstrip()
        return f"\\item {content_tex}"

    def visit_text(self, node: Text) -> str:
        return self._escape_tex(node.content)

    def visit_variable(self, node: Variable) -> str:
        # Look up variable in context
        value = self.context.get(node.name, f"${node.name}")

        # Convert value to TeX string
        if hasattr(value, "to_tex"):
            value_str = value.to_tex()
        elif hasattr(value, "to_string"):
            value_str = self._escape_tex(value.to_string())
        else:
            value_str = self._escape_tex(str(value))

        return value_str

    def visit_answer_blank(self, node: AnswerBlank) -> str:
        # Generate answer blank line for hardcopy
        width = node.width if node.width > 0 else 20
        # Approximate width in em units
        width_em = width * 0.5
        return f"\\underline{{\\hspace{{{width_em}em}}}}"

    def visit_code(self, node: Code) -> str:
        # Code execution placeholder (should be evaluated before TeX rendering)
        return "[code result]"

    def visit_math_inline(self, node: MathInline) -> str:
        # Use LaTeX inline math
        return f"${node.content}$"

    def visit_bold(self, node: Bold) -> str:
        content_tex = "".join(child.accept(self) for child in node.content)
        return f"\\textbf{{{content_tex}}}"

    def visit_italic(self, node: Italic) -> str:
        content_tex = "".join(child.accept(self) for child in node.content)
        return f"\\textit{{{content_tex}}}"

    # NEW FOR PARITY: TeX rendering for new node types

    def visit_heading(self, node: Any) -> str:
        """Render heading as TeX section commands."""
        from .parser import Heading
        if not isinstance(node, Heading):
            return ""

        content_tex = "".join(child.accept(self) for child in node.content)

        # Map heading levels to TeX commands
        heading_commands = {
            1: "section",
            2: "subsection",
            3: "subsubsection",
            4: "paragraph",
            5: "subparagraph",
            6: "subparagraph"
        }

        command = heading_commands.get(node.level, "paragraph")
        return f"\\{command}{{{content_tex}}}"

    def visit_rule(self, node: Any) -> str:
        """Render horizontal rule."""
        return "\\hrulefill"

    def visit_table(self, node: Any) -> str:
        """Render table in TeX."""
        from .parser import Table
        if not isinstance(node, Table):
            return ""

        if not node.rows:
            return ""

        # Determine column count from first row
        num_cols = len(node.rows[0].cells) if node.rows else 0
        col_spec = "l" * num_cols  # Left-aligned columns

        rows_tex = " \\\\\n".join(row.accept(self) for row in node.rows)
        return f"\\begin{{tabular}}{{{col_spec}}}\n{rows_tex}\n\\end{{tabular}}"

    def visit_table_row(self, node: Any) -> str:
        """Render table row in TeX."""
        from .parser import TableRow
        if not isinstance(node, TableRow):
            return ""

        cells_tex = []
        for cell_content in node.cells:
            cell_tex = "".join(child.accept(self) for child in cell_content)
            cells_tex.append(cell_tex)

        return " & ".join(cells_tex)

    def visit_align_block(self, node: Any) -> str:
        """Render alignment block."""
        from .parser import AlignBlock
        if not isinstance(node, AlignBlock):
            return ""

        content_tex = "".join(child.accept(self) for child in node.content)

        if node.alignment == "center":
            return f"\\begin{{center}}\n{content_tex}\n\\end{{center}}"
        elif node.alignment == "right":
            return f"\\begin{{flushright}}\n{content_tex}\n\\end{{flushright}}"
        elif node.alignment == "left":
            return f"\\begin{{flushleft}}\n{content_tex}\n\\end{{flushleft}}"
        else:
            return content_tex

    def visit_pre_block(self, node: Any) -> str:
        """Render pre-formatted block."""
        from .parser import PreBlock
        if not isinstance(node, PreBlock):
            return ""

        return f"\\begin{{verbatim}}\n{node.content}\n\\end{{verbatim}}"

    def visit_solution(self, node: Any) -> str:
        """Render solution section."""
        from .parser import Solution
        if not isinstance(node, Solution):
            return ""

        blocks_tex = "\n\n".join(block.accept(self) for block in node.content)
        return f"\\textbf{{Solution:}}\n\n{blocks_tex}"

    def visit_hint(self, node: Any) -> str:
        """Render hint section."""
        from .parser import Hint
        if not isinstance(node, Hint):
            return ""

        blocks_tex = "\n\n".join(block.accept(self) for block in node.content)
        return f"\\textbf{{Hint:}}\n\n{blocks_tex}"

    def _escape_tex(self, text: str) -> str:
        """Escape TeX special characters."""
        # Common TeX special characters
        replacements = {
            "\\": r"\textbackslash{}",
            "{": r"\{",
            "}": r"\}",
            "$": r"\$",
            "&": r"\&",
            "%": r"\%",
            "#": r"\#",
            "_": r"\_",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }

        result = text
        for char, replacement in replacements.items():
            result = result.replace(char, replacement)

        return result
