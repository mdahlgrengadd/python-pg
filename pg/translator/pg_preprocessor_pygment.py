"""
Rewritten PG Preprocessor to use Pygments and Lark

This module exposes a `PGPreprocessor` class compatible with the original
interface. It retains the logic for processing BEGIN_TEXT/BEGIN_PGML/etc.
blocks and do‑until loops, but uses a combination of Pygments and a
Lark grammar to rewrite Perl‑like code into Python.  The primary
motivation is to replace the brittle regex based line transformations
with a more structured approach.

REFACTORED ARCHITECTURE:
The preprocessor has been refactored into modular components for better
maintainability and testability. See REFACTORING_SUMMARY.md for details.

Extracted modules:
- pg_grammar.py: Lark grammar definition
- pg_transformer.py: AST transformation to IR
- pg_block_extractor.py: BEGIN_TEXT/PGML block extraction
- pg_text_processor.py: Text preprocessing (heredocs, interpolation)
- pg_pygments_rewriter.py: Fallback token-level rewriting
- pg_ir_emitter.py: IR to Python code generation

The high level flow of preprocessing is as follows:

1. Text preprocessing (heredocs, dereferences, string interpolation)
2. Split input into lines with line number mapping
3. Extract special blocks (BEGIN_TEXT/BEGIN_PGML/etc.) and store them
4. Parse remaining code with Lark grammar
5. Transform parse tree to intermediate representation (IR)
6. Emit Python code from IR
7. Fallback to Pygments rewriting for unparseable constructs
8. Join rewritten code with extracted blocks

The grammar covers common Perl constructs: assignments, declarations,
control flow, function calls, and PG-specific functions. It is intentionally
permissive—unknown constructs fall back to Pygments rewriting.

Note: This implementation requires `lark` and `pygments` packages.
Install via: pip install lark==1.1.5 pygments==2.18.0

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Tuple, Dict, Optional, Any

# Import modular components
from .pg_grammar import get_pg_grammar
from .pg_transformer import create_pg_transformer
from .pg_block_extractor import PGBlockExtractor
from .pg_text_processor import PGTextProcessor
from .pg_pygments_rewriter import PGPygmentsRewriter
from .pg_ir_emitter import PGIREmitter

# Import Pygments for token aware rewriting (kept for backward compatibility)
from pygments.lexers import get_lexer_by_name
from pygments.token import Token

# Try importing Lark for structured parsing.  If unavailable (for example
# in restricted environments), fall back to a no‑op parser.  This
# allows the preprocessor to operate purely on the Pygments fallback
# without raising ImportError.
try:
    # type: ignore[redefined-builtin]
    from lark import Lark, Transformer, v_args
    from lark.exceptions import LarkError  # type: ignore[misc]
    _LARK_AVAILABLE = True
except Exception:
    # Define dummy stand‑ins for the imported symbols
    _LARK_AVAILABLE = False

    class LarkError(Exception):
        pass

    def v_args(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

    class Transformer:
        pass
    # The Lark class is defined later in PGPreprocessor.__init__ when not available
    Lark = None  # type: ignore[assignment]


@dataclass
class PreprocessResult:
    """Result of preprocessing a PG file."""

    code: str
    """Preprocessed Python code"""

    text_blocks: List[Tuple[str, str]]
    """List of (block_type, content) for TEXT, PGML, SOLUTION, HINT blocks"""

    line_map: Dict[int, int]
    """Map from preprocessed line number to original line number"""


class PGPreprocessor:
    """
    Preprocess PG files to transform syntactic sugar into executable Python.

    This reimplementation uses Pygments and a Lark grammar instead of
    large regular expressions.  The public API remains identical to
    the original: the `preprocess` method returns a
    :class:`PreprocessResult` containing the generated code, the
    extracted text blocks, and a map from output lines to original
    source lines.

    REFACTORING NOTE:
    Key functionality has been extracted into separate modules for better
    maintainability. The following modules are now available for independent use:

    - pg_grammar: Grammar definition (get_pg_grammar())
    - pg_transformer: AST to IR transformation (create_pg_transformer())
    - pg_block_extractor: Block extraction (PGBlockExtractor)
    - pg_text_processor: Text preprocessing (PGTextProcessor)
    - pg_pygments_rewriter: Fallback rewriting (PGPygmentsRewriter)
    - pg_ir_emitter: Code generation (PGIREmitter)

    These modules can be used independently for testing, customization, or
    building alternative preprocessors. See REFACTORING_SUMMARY.md for details.
    """

    # Block markers used to detect special PG sections
    BLOCK_PATTERNS = {
        "TEXT": (r"BEGIN_TEXT\s*$", r"^END_TEXT"),
        "PGML": (r"BEGIN_PGML\s*$", r"^END_PGML"),
        "SOLUTION": (r"BEGIN_SOLUTION\s*$", r"^END_SOLUTION"),
        "HINT": (r"BEGIN_HINT\s*$", r"^END_HINT"),
        "PGML_SOLUTION": (r"BEGIN_PGML_SOLUTION\s*$", r"^END_PGML_SOLUTION"),
        "PGML_HINT": (r"BEGIN_PGML_HINT\s*$", r"^END_PGML_HINT"),
        "TIKZ": (r"BEGIN_TIKZ\s*$", r"^END_TIKZ"),
    }

    def __init__(self) -> None:
        # Initialize modular components
        self._text_processor = PGTextProcessor()
        self._block_extractor = PGBlockExtractor()
        self._pygments_rewriter = PGPygmentsRewriter()

        # Initialise Pygments lexer once (kept for backward compatibility in some methods)
        self._perl_lexer = get_lexer_by_name("perl")

        # Closure counter for generating unique function names
        self._closure_counter = 0

        # Attempt to prepare the Lark parser.  In restricted
        # environments where Lark is unavailable, the parser will be
        # left as None and the preprocessor will rely on manual
        # pattern matching and the Pygments fallback exclusively.
        if _LARK_AVAILABLE:
            # Use the Earley parser for our grammar to avoid reduce/reduce
            # conflicts that arise when calls appear both as statements and
            # as subexpressions.  The Earley parser is slower but more
            # forgiving for the relatively small code fragments we parse
            # here.  We still enable propagate_positions so that line
            # mapping works when we lower expressions via Lark.
            self._parser = Lark(
                get_pg_grammar(),  # Use the extracted grammar
                start="start",
                parser="earley",
                maybe_placeholders=True,
                propagate_positions=True,
            )
            self._transformer = create_pg_transformer()  # Use the extracted transformer
            # Initialize IR emitter with pygments rewriter for fallback
            self._ir_emitter = PGIREmitter(self._pygments_rewriter)
        else:
            self._parser = None
            self._transformer = None
            self._ir_emitter = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, pg_source: str, use_sandbox_macros: bool = False, standalone: bool = False) -> PreprocessResult:
        """
        Preprocess PG source code.

        Args:
            pg_source: Raw PG file content
            use_sandbox_macros: If True, assumes sandbox pre-loads macros (no imports generated)
            standalone: If True, generates standalone executable .pyg with boilerplate

        Returns:
            PreprocessResult with transformed code and metadata
        """
        # Convert Perl heredocs (<<END_MARKER) to Python syntax BEFORE splitting into lines
        pg_source = self._text_processor.convert_heredocs_global(pg_source)

        # Fix reference dereference arrows before parsing
        # $x->[$i] -> $x[i], $x->{key} -> $x[key]
        pg_source = self._text_processor.fix_reference_dereferences(pg_source)

        # Note: Don't convert ->with( early! The Lark grammar treats . as a binary operator,
        # so if we convert ->with( to .with_params(, Lark will parse it as string concatenation.
        # Instead, we'll handle ->with( in the IR emission phase via post-processing.

        # Handle Perl's . (concatenation) and x (repetition) operators for compatibility with Vector dot/cross
        # These will be converted to pg_concat/pg_repeat which can handle both string ops and Vector operations
        # Use word boundaries to avoid matching dots in numbers or method calls
        import re
        # Replace ` . ` with ` pg_concat( ... ) ` - matches spaces around the dot
        # Handle both $var and var (with and without leading $)
        pg_source = re.sub(r'(\$?\w+)\s+\.\s+(\$?\w+)',
                           r'pg_concat(\1, \2)', pg_source)
        # Replace ` x ` with ` pg_repeat( ... ) ` - matches word boundaries around x operator
        # But be careful not to match 'x' in identifiers, so require word boundaries
        pg_source = re.sub(r'(\$?\w+)\s+x\s+(\$?\w+)',
                           r'pg_repeat(\1, \2)', pg_source)

        lines = pg_source.split("\n")
        output_lines: List[str] = []
        text_blocks: List[Tuple[str, str]] = []
        line_map: Dict[int, int] = {}

        # Collect import lines from loadMacros when not using sandbox macros
        import_lines: List[str] = []
        loaded_macros_comment: Optional[str] = None
        if not use_sandbox_macros:
            # If standalone mode, always import MathObjects
            # (Context, Compute, etc. are used in almost every problem)
            if standalone:
                import_lines.append("from pg.mathobjects import *")

            for line in lines:
                if "loadMacros" in line:
                    match = re.search(r'loadMacros\((.*?)\)', line, re.DOTALL)
                    if match:
                        imports, comment = self._transform_load_macros(
                            match.group(1))
                        import_lines.extend(imports)
                        loaded_macros_comment = comment

        imports_inserted = False
        in_load_macros = False
        paren_depth = 0

        i = 0
        while i < len(lines):
            original_line = lines[i]
            line_start_index = i
            stripped_line = original_line.strip()
            if stripped_line.startswith('package '):
                i += 1
                continue
            if stripped_line.startswith('our') and 'ISA' in stripped_line:
                i += 1
                continue

            # Join Perl-style implicit continuations to make downstream parsing easier
            is_comment = original_line.lstrip(' \t').startswith('#')
            # Check if this is a special block marker (BEGIN_PGML, BEGIN_TEXT, etc.)
            is_special_block = any(
                re.match(pattern, original_line.strip())
                for pattern, _ in self.BLOCK_PATTERNS.values()
            )
            if not is_comment and not is_special_block:
                if not re.match(r'^\s*\}\s*else\s*\{\s*$', original_line):
                    while i + 1 < len(lines):
                        stripped = original_line.rstrip()
                        next_line = lines[i + 1]

                        # Stop joining if the next line is a comment or a } else { boundary
                        if next_line.lstrip(' \t').startswith('#'):
                            break
                        if re.match(r'^\s*\}\s*else\s*\{\s*$', next_line):
                            break

                        should_join = False
                        stripped_no_comment = self._text_processor.strip_inline_comment(
                            stripped)
                        # Don't join if the line ends with a semicolon (complete statement)
                        if stripped_no_comment and stripped_no_comment[-1] == ';':
                            should_join = False
                        elif stripped_no_comment and stripped_no_comment[-1] in '=,([':
                            if next_line and next_line[0] in ' \t':
                                should_join = True

                        if not should_join:
                            next_stripped = next_line.lstrip(' \t')
                            if next_stripped and next_stripped[0] in '.+-':
                                if next_stripped[0] == '.' or (
                                    next_stripped[0] in '+-'
                                    and len(next_stripped) > 1
                                    and next_stripped[1] in ' \t"\''
                                ):
                                    should_join = True

                        if not should_join:
                            next_stripped = next_line.lstrip(' \t')
                            if next_stripped and next_stripped.startswith('->'):
                                should_join = True

                        if not should_join:
                            # Special case: if line ends with } and next line starts with 'until', we should join
                            # Also, keep joining if we have logical operators (&&, ||) or continue until we find semicolon
                            next_stripped = next_line.lstrip(' \t')
                            if (stripped and stripped.endswith('}') and
                                    next_stripped and next_stripped.startswith('until')):
                                should_join = True
                            # Special case: if current line contains 'map {' or 'grep {' with balanced braces,
                            # the next line is the iterable, so join them
                            elif (stripped and re.search(r'(map|grep)\s*\{', stripped) and
                                  stripped.endswith('}') and
                                  next_stripped):
                                should_join = True
                            # If current line has logical operators, keep joining
                            elif (stripped and (stripped.endswith('&&') or stripped.endswith('||') or
                                                next_stripped.startswith('&&') or next_stripped.startswith('||'))):
                                if next_line.lstrip(' \t'):  # Not empty
                                    should_join = True

                        if not should_join:
                            check_line = self._text_processor.strip_inline_comment(stripped)
                            # Account for $# operator which shouldn't affect paren counting
                            # Replace $#name with a placeholder
                            check_line_adjusted = re.sub(
                                r'\$#\w+', '_placeholder_', check_line)
                            open_count = (
                                check_line_adjusted.count('(')
                                + check_line_adjusted.count('[')
                                + check_line_adjusted.count('{')
                            )
                            close_count = (
                                check_line_adjusted.count(')')
                                + check_line_adjusted.count(']')
                                + check_line_adjusted.count('}')
                            )
                            if open_count > close_count:
                                should_join = True

                        if should_join and next_line.strip():
                            line_without_comment = self._text_processor.strip_inline_comment(
                                original_line.rstrip()
                            )
                            original_line = (
                                line_without_comment + ' ' +
                                next_line.lstrip(' \t')
                            )
                            i += 1
                        else:
                            break

            output_line_num = len(output_lines) + 1
            line_map[output_line_num] = line_start_index + 1

            # Convert map/grep blocks to Python list comprehensions BEFORE parsing
            original_line = self._convert_map_grep_blocks(original_line)

            # Handle compound lines with loadMacros
            if ';' in original_line and 'loadMacros' in original_line:
                parts = original_line.split(';')
                non_loadmacros_parts: List[str] = []
                for part in parts:
                    part = part.strip()
                    if 'loadMacros' in part:
                        # Determine if this loadMacros is complete on one line
                        if '(' in part and part.count('(') == part.count(')'):
                            # Single line loadMacros: skip it
                            continue
                        else:
                            # Multi line loadMacros: start skipping
                            in_load_macros = True
                            paren_depth = part.count('(') - part.count(')')
                            break
                    else:
                        if part:
                            non_loadmacros_parts.append(part)
                if non_loadmacros_parts:
                    combined = '; '.join(non_loadmacros_parts)
                    if combined:
                        compiled_lines = self._compile_line(combined)
                        output_lines.extend(compiled_lines)
                i += 1
                continue

            # Standalone loadMacros start
            if 'loadMacros' in original_line and '(' in original_line and not in_load_macros:
                in_load_macros = True
                paren_depth = original_line.count(
                    '(') - original_line.count(')')
                if paren_depth == 0:
                    in_load_macros = False
                i += 1
                continue

            # If inside loadMacros, skip lines until parens balanced
            if in_load_macros:
                paren_depth += original_line.count('(') - \
                    original_line.count(')')
                if paren_depth <= 0:
                    in_load_macros = False
                i += 1
                continue

            # Insert imports before/after DOCUMENT() depending on standalone mode
            if not imports_inserted and re.match(r'^\s*DOCUMENT\(\s*\)', original_line):
                # For standalone mode, emit imports BEFORE DOCUMENT()
                if standalone and import_lines:
                    output_lines.extend(import_lines)
                    if loaded_macros_comment:
                        output_lines.append(loaded_macros_comment)
                    output_lines.append("")

                # Emit DOCUMENT() line
                # Split multiple statements by semicolon and handle individually
                if ';' in original_line:
                    parts = [p.strip()
                             for p in original_line.split(';') if p.strip()]
                    for part in parts:
                        if 'loadMacros' in part:
                            continue
                        compiled_lines = self._compile_line(part)
                        output_lines.extend(compiled_lines)
                else:
                    compiled_lines = self._compile_line(original_line)
                    output_lines.extend(compiled_lines)

                # For normal mode, emit imports AFTER DOCUMENT()
                if not standalone and import_lines:
                    output_lines.append("")
                    output_lines.extend(import_lines)
                    if loaded_macros_comment:
                        output_lines.append(loaded_macros_comment)
                    output_lines.append("")

                imports_inserted = True
                i += 1
                continue

            # Handle standalone Perl subroutine definitions: sub name { ... }
            # These are typically used in packages for method overrides
            standalone_sub_match = re.match(
                r'^\s*sub\s+(\w+)\s*\{', original_line)
            if standalone_sub_match:
                # Collect the subroutine body until we find the matching closing brace
                sub_lines = [original_line]
                brace_depth = original_line.count(
                    '{') - original_line.count('}')
                i += 1
                max_sub_lines = 100
                lines_collected = 0
                while i < len(lines) and brace_depth > 0 and lines_collected < max_sub_lines:
                    current_line = lines[i]
                    sub_lines.append(current_line)
                    brace_depth += current_line.count(
                        '{') - current_line.count('}')
                    i += 1
                    lines_collected += 1

                # Comment out the entire subroutine definition
                # (It's probably defining a method override that won't work in Python anyway)
                for sub_line in sub_lines:
                    output_lines.append(
                        f"# {sub_line}  # Perl subroutine definition skipped")
                continue

            # Parse and translate Perl sub { ... } closures using Lark grammar
            sub_match = re.search(r'(=>|=)\s*sub\s*\{', original_line)
            if sub_match:
                # Collect the closure lines (same multi-line logic as before)
                sub_start = original_line.find('sub {')
                after_sub = original_line[sub_start + 5:]
                single_line_brace_count = after_sub.count(
                    '{') - after_sub.count('}')

                closure_lines = [original_line]
                brace_depth = 1 + single_line_brace_count
                max_closure_lines = 100
                lines_collected = 0
                if brace_depth > 0:
                    i += 1
                    while i < len(lines) and brace_depth > 0 and lines_collected < max_closure_lines:
                        current_line = lines[i]
                        closure_lines.append(current_line)
                        brace_depth += current_line.count(
                            '{') - current_line.count('}')
                        i += 1
                        lines_collected += 1
                else:
                    i += 1

                if brace_depth > 0 and lines_collected >= max_closure_lines:
                    output_lines.append(
                        f"# {original_line[:80]}... # Closure too complex, skipped")
                    continue

                first_line = closure_lines[0]
                last_line = closure_lines[-1] if closure_lines else ''

                # Extract prefix and suffix around the closure
                sub_start = first_line.find('sub {')
                prefix = first_line[:sub_start]

                # Find closing brace and suffix
                if len(closure_lines) == 1:
                    # Single-line closure
                    brace_count = 1
                    pos = sub_start + 5
                    while pos < len(first_line) and brace_count > 0:
                        if first_line[pos] == '{':
                            brace_count += 1
                        elif first_line[pos] == '}':
                            brace_count -= 1
                        pos += 1
                    suffix_from_last_line = first_line[pos:] if pos < len(
                        first_line) else ''
                else:
                    # Multi-line closure
                    close_brace_match = re.search(r'\}(.*)$', last_line)
                    suffix_from_last_line = close_brace_match.group(
                        1) if close_brace_match else ''

                # Join closure lines and extract just the sub { ... } part
                closure_text = '\n'.join(closure_lines)
                sub_body_start = closure_text.find('sub {')
                if sub_body_start < 0:
                    output_lines.append(
                        f"# {first_line}  # Could not find closure")
                    continue

                # Try to parse and translate with Lark
                try:
                    # Extract the closure expression: sub { ... }
                    sub_start_pos = sub_body_start
                    brace_count = 1
                    pos = sub_body_start + 5  # After 'sub {'
                    while pos < len(closure_text) and brace_count > 0:
                        if closure_text[pos] == '{':
                            brace_count += 1
                        elif closure_text[pos] == '}':
                            brace_count -= 1
                        pos += 1

                    # Includes 'sub { ... }'
                    closure_expr_text = closure_text[sub_start_pos:pos]

                    if self._parser is not None:
                        # Try to parse the closure as an expression within a dummy assignment
                        # This allows us to use the normal 'start' rule instead of a special 'sub_closure' rule
                        # Note: must use $dummy (with sigil) because grammar only accepts variables with sigils

                        # Undo early transformations that were applied to the whole source
                        # before we had a chance to parse the closure
                        # The grammar expects @$var syntax, not list($var)
                        closure_for_parse = closure_expr_text
                        closure_for_parse = re.sub(
                            r'list\(\$(\w+)\)', r'@$\1', closure_for_parse)

                        # Clean up excessive whitespace but preserve structure
                        # Replace multiple spaces/tabs with single space, keep newlines for readability in errors
                        cleaned_closure = ' '.join(closure_for_parse.split())
                        dummy_stmt = f"$__closure__ = {cleaned_closure};"
                        try:
                            tree = self._parser.parse(dummy_stmt)
                            # Extract the assignment IR from the parse tree
                            stmt_list = self._transformer.transform(tree)
                            if stmt_list and len(stmt_list) > 0:
                                assign_ir = stmt_list[0]
                                # assign_ir should be ("assign", var, closure_ir)
                                if isinstance(assign_ir, tuple) and assign_ir[0] == "assign" and len(assign_ir) >= 3:
                                    # Extract the RHS (the closure)
                                    closure_ir = assign_ir[2]
                                else:
                                    closure_ir = None
                            else:
                                closure_ir = None
                        except Exception as parse_err:
                            # Parsing as statement failed, try alternative approach
                            # The closure may contain unsupported Perl constructs or edge cases
                            closure_ir = None

                        # Emit the closure to Python
                        if closure_ir and closure_ir[0] == "closure":
                            # Extract context name for better function naming
                            context_match = re.search(
                                r'(\w+)\s*(?:=>|=)\s*sub\s*\{', first_line)
                            context_name = context_match.group(
                                1) if context_match else "closure"

                            # Calculate current indentation level
                            indent_match = re.match(r'^(\s*)', prefix)
                            indent_str = indent_match.group(
                                1) if indent_match else ''
                            current_indent = len(indent_str) // 4

                            # Emit closure directly with context name (may return tuple for complex closures)
                            _, body_stmts = closure_ir
                            closure_result = self._emit_closure(
                                body_stmts, current_indent, context_name)
                            closure_py = None

                            if isinstance(closure_result, tuple) and len(closure_result) == 2:
                                # Complex closure - extracted to function
                                func_def_lines, func_ref = closure_result
                                output_lines.extend(func_def_lines)
                                # Blank line for readability
                                output_lines.append("")
                                closure_py = func_ref
                            else:
                                # Simple closure - inline lambda
                                closure_py = closure_result
                        else:
                            closure_py = None

                        if closure_py:
                            # Reconstruct the line
                            transformed_line = f"{prefix}{closure_py}{suffix_from_last_line}"
                            # Further transform (=> to =, etc.)
                            final_line = self._rewrite_statement(
                                transformed_line)
                            if final_line:
                                output_lines.append(final_line)
                            else:
                                output_lines.append(transformed_line)
                            continue  # Successfully translated, skip to next closure

                        # If we get here, parsing or emission failed - fall through to fallback
                        # Don't use try-except, just check if we can stub it
                        param_match = re.search(
                            r'(\w+)\s*=>\s*sub\s*\{', first_line)
                        if param_match:
                            stubbed_line = f"{prefix}lambda *args, **kwargs: None{suffix_from_last_line}"
                            transformed = self._rewrite_statement(stubbed_line)
                            if transformed:
                                output_lines.append(
                                    transformed + "  # Stubbed Perl closure (parsing failed)")
                        else:
                            assign_match = re.search(
                                r'(\w+)\s*=\s*sub\s*\{', first_line)
                            if assign_match:
                                var_name = assign_match.group(1)
                                indent_match = re.match(r'^(\s*)', first_line)
                                indent = indent_match.group(
                                    1) if indent_match else ''
                                stubbed_line = f"{indent}{var_name} = lambda *args, **kwargs: None"
                                transformed = self._rewrite_statement(
                                    stubbed_line)
                                if transformed:
                                    output_lines.append(
                                        transformed + "  # Stubbed Perl closure (parsing failed)")
                            else:
                                output_lines.append(
                                    f"# {first_line}  # Skipped Perl closure")
                    else:
                        output_lines.append(
                            f"# {first_line}  # Parser not available")

                except Exception as e:
                    # Fall back to stubbing if Lark parsing fails
                    # This maintains backward compatibility for complex or unsupported closures
                    param_match = re.search(
                        r'(\w+)\s*=>\s*sub\s*\{', first_line)
                    if param_match:
                        stubbed_line = f"{prefix}lambda *args, **kwargs: None{suffix_from_last_line}"
                        transformed = self._rewrite_statement(stubbed_line)
                        if transformed:
                            output_lines.append(
                                transformed + "  # Stubbed Perl closure (parsing failed)")
                    else:
                        assign_match = re.search(
                            r'(\w+)\s*=\s*sub\s*\{', first_line)
                        if assign_match:
                            var_name = assign_match.group(1)
                            indent_match = re.match(r'^(\s*)', first_line)
                            indent = indent_match.group(
                                1) if indent_match else ''
                            stubbed_line = f"{indent}{var_name} = lambda *args, **kwargs: None"
                            transformed = self._rewrite_statement(stubbed_line)
                            if transformed:
                                output_lines.append(
                                    transformed + "  # Stubbed Perl closure (parsing failed)")
                        else:
                            output_lines.append(
                                f"# {first_line}  # Skipped Perl closure")
                continue

            # Detect do { ... } until loops (single or multi line)
            # This regex handles:
            # - do { body } until (condition)
            # - do { body } until (cond1) && (cond2) && (cond3);
            # It captures everything from 'until' to the end of the line/statement
            do_until_single = re.match(
                r'^\s*do\s*\{([^}]*)\}\s*until\s+(.+)$', original_line)
            if do_until_single:
                body = do_until_single.group(1).strip()
                full_condition = do_until_single.group(2).strip()
                # Remove trailing semicolon if present
                if full_condition.endswith(';'):
                    full_condition = full_condition[:-1].strip()
                # Extract the actual condition (may be wrapped in parens or have logical operators)
                # If it starts with ( and ends with ), extract the content
                if full_condition.startswith('(') and full_condition.endswith(')'):
                    condition = full_condition[1:-1].strip()
                else:
                    # Otherwise use the whole thing
                    condition = full_condition
                # Compile body and condition via grammar or fallback
                body_lines = self._compile_line(body)
                # Flatten to a single statement; indent body lines
                compiled_cond = self._compile_expr(condition)
                # Strip outer parens if already present
                if compiled_cond.startswith('(') and compiled_cond.endswith(')'):
                    compiled_cond = compiled_cond[1:-1]
                # do-until means: execute body, then repeat UNTIL condition is true
                # In Python: while True: body; if condition: break
                output_lines.append(f'while True:')
                for bl in body_lines:
                    output_lines.append('    ' + bl)
                output_lines.append(f'    if ({compiled_cond}):')
                output_lines.append('        break')
                i += 1
                continue
            # Multi line do until
            do_until_start = re.match(r'^\s*do\s*\{', original_line)
            if do_until_start:
                block_lines = [original_line]
                brace_depth = original_line.count(
                    '{') - original_line.count('}')
                i += 1

                # Collect lines until braces are balanced
                while i < len(lines) and brace_depth > 0:
                    ln = lines[i]
                    block_lines.append(ln)
                    brace_depth += ln.count('{') - ln.count('}')
                    i += 1

                # After braces are balanced, check if there's an 'until' on the same line as '}'
                last_line = block_lines[-1] if block_lines else ""
                until_match = re.search(r'\}\s*until\s*\(([^)]+)\)', last_line)

                # If 'until' is not on the closing brace line, check the next line
                if not until_match and i < len(lines):
                    next_line = lines[i]
                    until_match_next = re.match(r'^\s*until\s+', next_line)
                    if until_match_next:
                        # Found 'until' on the next line, need to collect the full condition
                        # which may span multiple lines (e.g., until (...) && (...) && (...);)
                        condition_lines = [next_line]
                        condition_line_idx = i
                        i += 1

                        # Keep collecting lines until we find a semicolon that ends the statement
                        while i < len(lines) and ';' not in condition_lines[-1]:
                            ln = lines[i]
                            condition_lines.append(ln)
                            i += 1

                        # Concatenate all condition lines
                        full_condition = ' '.join(ln.strip()
                                                  for ln in condition_lines)
                        # Extract the condition from "until ... ;"
                        condition_match = re.match(
                            r'^\s*until\s+(.+?);?\s*$', full_condition)
                        if condition_match:
                            condition_raw = condition_match.group(1).strip()
                            # Remove surrounding parens if present
                            if condition_raw.startswith('(') and condition_raw.endswith(')'):
                                condition = condition_raw[1:-1].strip()
                            else:
                                condition = condition_raw
                            until_match = True  # Mark as found
                            # DO NOT add condition lines to block_lines!
                            # They are part of the 'until' clause, not the 'do { }' body
                        else:
                            until_match = None
                    else:
                        until_match = None

                if until_match:
                    if isinstance(until_match, bool):
                        # Already extracted condition from next line
                        pass
                    else:
                        # Extract from same-line pattern
                        condition = until_match.group(1).strip()
                    # Extract body lines: remove 'do {' and '} until (...)'
                    inner_lines: List[str] = []

                    # All lines in block_lines are now part of the body (not the condition)
                    # since we don't add condition lines to block_lines anymore

                    if len(block_lines) == 1 and block_lines[0].count('{') == block_lines[0].count('}'):
                        # Extract content between 'do {' and '}'
                        body_match = re.match(
                            r'^\s*do\s*\{(.*)\}\s*$', block_lines[0])
                        if body_match:
                            body_content = body_match.group(1).strip()
                            if body_content:
                                inner_lines.append(body_content)
                    else:
                        # Multi-line case: shouldn't happen since we only collect until brace_depth > 0
                        # But handle it just in case
                        # Remove the first line's 'do {'
                        first_body = re.sub(
                            r'^\s*do\s*\{', '', block_lines[0]).strip()
                        if first_body:
                            inner_lines.append(first_body)

                        # Middle lines
                        body_end_idx = len(block_lines) - 1

                        for middle_idx in range(1, body_end_idx):
                            inner_lines.append(block_lines[middle_idx])

                        # Process the line with the closing brace (if it's not the first line)
                        if body_end_idx > 0:
                            last_body_line = block_lines[body_end_idx]
                            # Remove the closing }
                            last_body = re.sub(
                                r'\}\s*$', '', last_body_line).strip()
                            if last_body:
                                inner_lines.append(last_body)

                    # Compile body lines
                    compiled_body: List[str] = []
                    for ln in inner_lines:
                        compiled_body.extend(self._compile_line(ln))
                    compiled_cond = self._compile_expr(condition)
                    # Strip outer parens if already present
                    if compiled_cond.startswith('(') and compiled_cond.endswith(')'):
                        compiled_cond = compiled_cond[1:-1]
                    # Emit Python loop: do-until means execute body, then repeat UNTIL condition is true
                    # In Python: while True: body; if condition: break
                    output_lines.append(f'while True:')
                    for cb in compiled_body:
                        output_lines.append('    ' + cb)
                    output_lines.append(f'    if ({compiled_cond}):')
                    output_lines.append('        break')
                    continue
                else:
                    # Not a proper do-until, fall through: rewind index to process lines normally
                    i = i - len(block_lines) + 1

            # Detect Perl for/foreach loops (do this BEFORE block detection)
            # so that blocks inside the loop are handled as part of the loop compilation
            for_result = self._try_rewrite_for_loop(lines, line_start_index)
            if for_result is not None:
                rewritten_loop, consumed_lines = for_result
                output_lines.extend(rewritten_loop)
                i = line_start_index + consumed_lines
                continue

            # Block detection: check for BEGIN_* markers
            block_found = False
            for block_type, (begin_pattern, end_pattern) in self.BLOCK_PATTERNS.items():
                # Skip method-call style TikZ/LaTeX (handled separately below)
                # Updated to handle array indexing like $graph[$i]
                if re.search(begin_pattern, original_line) and not re.search(r'\$\w+(?:\[[^\]]*\])?\s*->\s*(BEGIN_TIKZ|BEGIN_LATEX_IMAGE)', original_line):
                    block_content_lines: List[str] = []
                    i += 1
                    while i < len(lines) and not re.match(end_pattern, lines[i]):
                        block_content_lines.append(lines[i])
                        i += 1
                    block_content = "\n".join(block_content_lines)
                    text_blocks.append((block_type, block_content))
                    # PGML blocks require evaluator transformation before storing
                    if "PGML" in block_type:
                        transformed_pgml = self._block_extractor.transform_pgml_evaluators(
                            block_content)
                        block_var = f"PGML_BLOCK_{len(text_blocks) - 1}"
                        escaped_content = self._block_extractor.escape_triple_quotes(
                            transformed_pgml)
                        output_lines.append(
                            f"{block_var} = '''\n{escaped_content}\n'''")
                        if "SOLUTION" in block_type:
                            output_lines.append(f"SOLUTION(PGML({block_var}))")
                        elif "HINT" in block_type:
                            output_lines.append(f"HINT(PGML({block_var}))")
                        else:
                            output_lines.append(f"TEXT(PGML({block_var}))")
                    elif block_type == "TIKZ":
                        # Preserve raw TikZ/TeX content verbatim in a raw string
                        block_var = f"TIKZ_BLOCK_{len(text_blocks) - 1}"
                        escaped_content = block_content.replace(
                            "'''", r"\'\'\'")
                        # Use raw string (r'...') to preserve backslashes in TikZ code
                        output_lines.append(f"{block_var} = r'''")
                        output_lines.append(escaped_content)
                        output_lines.append("'''")
                    else:
                        transformed_content = self._text_processor.transform_text_block(
                            block_content)
                        if "SOLUTION" in block_type:
                            output_lines.append(
                                f"SOLUTION({transformed_content})")
                        elif "HINT" in block_type:
                            output_lines.append(f"HINT({transformed_content})")
                        else:
                            output_lines.append(f"TEXT({transformed_content})")
                    block_found = True
                    break
            if block_found:
                # Skip over the END marker by incrementing i once more
                i += 1
                continue

            # Handle method-call-style blocks: $obj->BEGIN_TIKZ or $obj->BEGIN_LATEX_IMAGE
            # These should capture content until END_TIKZ/END_LATEX_IMAGE and pass as raw string
            # Note: Use search to find these patterns even if there's trailing whitespace/comments
            # Also handle array indexing like $graph[$i]->BEGIN_TIKZ
            tikz_method_match = re.search(
                r'(\$\w+(?:\[[^\]]*\])*)\s*->\s*BEGIN_TIKZ', original_line)
            latex_method_match = re.search(
                r'(\$\w+(?:\[[^\]]*\])*)\s*->\s*BEGIN_LATEX_IMAGE', original_line)

            if (tikz_method_match or latex_method_match) and original_line.strip().endswith(('BEGIN_TIKZ', 'BEGIN_LATEX_IMAGE')):
                obj_var = (tikz_method_match or latex_method_match).group(1)
                end_marker = "END_TIKZ" if tikz_method_match else "END_LATEX_IMAGE"
                method_name = "BEGIN_TIKZ" if tikz_method_match else "BEGIN_LATEX_IMAGE"

                # Collect content lines until we hit the end marker
                content_lines: List[str] = []
                i += 1
                while i < len(lines):
                    if re.match(rf'^\s*{end_marker}\s*$', lines[i]):
                        break
                    content_lines.append(lines[i])
                    i += 1

                # Join content and escape for raw string
                content = '\n'.join(content_lines)
                # Use raw string to avoid backslash issues
                escaped_content = content.replace("'''", r"\'\'\'")

                # Convert $obj_var to Python name (remove $)
                py_var = obj_var[1:] if obj_var.startswith('$') else obj_var

                # Preserve indentation from the original line
                match = re.match(r'^(\s*)', original_line)
                indent = match.group(1) if match else ''

                # Emit the method call with content as argument
                # Split into multiple lines to avoid embedding newlines in f-string
                output_lines.append(f"{indent}{py_var}.{method_name}(r'''")
                output_lines.append(escaped_content)
                output_lines.append(f"{indent}''')")

                # Skip the END marker
                i += 1
                continue

            # Otherwise compile the current line
            compiled_lines = self._compile_line(original_line)
            output_lines.extend(compiled_lines)
            i += 1

        code = "\n".join(output_lines)
        # Post-process to initialize arrays/dicts that are assigned to without declaration
        code = self._initialize_arrays(code)

        # Handle Perl array slice assignments: @inversion[@shuffle] = (0 .. $#shuffle)
        # This creates an inverted mapping: inversion[shuffle[i]] = i
        # Pattern: dict_var[list_var] = range_expr or (range_expr) or (list with items)
        # Use non-greedy match to handle nested parentheses in range args
        code = re.sub(
            r'^(\s*)([a-z_]\w*)\[([a-z_]\w*)\]\s*=\s*\(?range\(.*?\)\)?\s*$',
            lambda m: f"{m.group(1)}for __i, __idx in enumerate({m.group(3)}):\n{m.group(1)}    {m.group(2)}[__idx] = __i",
            code,
            flags=re.MULTILINE
        )

        # Convert empty tuple assignments to empty lists for array variables
        # In Perl: @var = () creates an empty list
        # In Python: () is a tuple, [] is a list, so we need to convert
        code = re.sub(r'^\s*([a-z_]\w*)\s*=\s*\(\)\s*$',
                      r'\1 = []', code, flags=re.MULTILINE)

        # Fix array slices with range: array[range(a, b)] -> array[a:b]
        # This happens when Perl @array[0..$#array] is converted to array[range(0, len(array))]
        # Need to handle nested parentheses in expressions like len(answers)
        # Use non-greedy matching to correctly capture arguments
        code = re.sub(
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\[\s*range\s*\(\s*([^,]+?)\s*,\s*(.*?)\s*\)\s*\]',
            lambda m: f"{m.group(1)}[{m.group(2).strip()}:{m.group(3).strip()}]",
            code
        )

        # Convert parenthesized list assignments to list literals
        # Pattern: var = (item1, item2, ...) or var = (single_item,)
        # This handles Perl array assignments like @seq = (1, 1)
        # After sigil removal: seq = (1, 1) should become seq = [1, 1]
        # We need to be careful not to convert function calls like foo = func(a, b)
        # Match: assignment where RHS is a parenthesized list with commas
        code = re.sub(
            r'^(\s*[a-z_]\w*)\s*=\s*\(([^)]+,[^)]*)\)\s*$',
            r'\1 = [\2]',
            code,
            flags=re.MULTILINE
        )

        # Also convert parenthesized single expressions that are ranges to lists
        # Pattern: var = (range(...)) should become var = list(range(...))
        # This handles cases like: indices = (0 .. $#answers)
        code = re.sub(
            r'^(\s*[a-z_]\w*)\s*=\s*\(\s*range\s*\((.+)\)\s*\)\s*$',
            r'\1 = list(range(\2))',
            code,
            flags=re.MULTILINE
        )

        # TODO: Convert implicit multiplication in quoted strings
        # Pattern: in strings like 'sin(x)' or '2pi', add explicit * for implicit multiplication
        # This is complex because we need to avoid converting things like 'html' or variable names
        # Examples that need conversion: '2pi' -> '2*pi', '5sin(x)' -> '5*sin(x)'
        # But should NOT convert: 'abc', 'sin', 'pi' alone
        # Requires careful heuristics or deeper parsing

        # Convert postfix for/foreach loops to regular for loops
        # Pattern: statement for iterable  ->  for var in iterable: statement
        # Most common: var += expr for iterable
        def convert_postfix_for(match):
            """Convert postfix for loop to regular for loop."""
            indent = match.group(1)
            statement = match.group(2).strip()
            # Extract loop variable from statement (usually $_)
            # For += patterns: capture the variable and iterable
            add_match = re.match(
                r'(\w+)\s*\+=\s*(.+?)\s+for\s+(.+)', statement)
            if add_match:
                var = add_match.group(1)
                expr = add_match.group(2)
                iterable = add_match.group(3)
                # Convert to:
                # for _ in iterable:
                #     var += _
                # Use _ as the implicit loop variable in Perl
                return f"{indent}for _ in {iterable}:\n{indent}    {var} += _"
            return match.group(0)

        code = re.sub(
            r'^(\s*)(\w+\s*\+=\s*.+?\s+for\s+.+?)$',
            convert_postfix_for,
            code,
            flags=re.MULTILINE
        )

        # Fix _.[...] pattern (shouldn't have a dot before bracket in Python)
        # This occurs when $_ -> [...] is converted to _ . [...]
        # In Python, we just want _[...]
        code = re.sub(r'\b_\.(\[)', r'_\1', code)

        # If standalone mode, add boilerplate for direct execution
        if standalone:
            boilerplate = '''

if __name__ == "__main__":
    """Execute problem and display results."""
    from pg.macros.core.pg_core import get_environment
    
    env = get_environment()
    if env:
        print("=" * 80)
        print("PROBLEM STATEMENT")
        print("=" * 80)
        print(''.join(env.output_array))
        
        if env.solution_array:
            print("\\n" + "=" * 80)
            print("SOLUTION")
            print("=" * 80)
            print(''.join(env.solution_array))
        
        if env.hint_array:
            print("\\n" + "=" * 80)
            print("HINT")
            print("=" * 80)
            print(''.join(env.hint_array))
        
        print("\\n" + "=" * 80)
        print(f"ANSWERS: {len(env.answers_hash)} answer blank(s)")
        print("=" * 80)
        for name in sorted(env.answers_hash.keys()):
            print(f"  {name}")
'''
            code += boilerplate

        # Postprocessing: Convert list literals to PerlList() for Perl-like array behavior
        # This allows arrays to auto-vivify when assigning to arbitrary indices
        import re as re_module

        # Match: var = [...] and wrap with PerlList()
        # We need to handle nested brackets properly
        def wrap_with_perllist_nested(code_str):
            """Convert list literals to PerlList, handling nested brackets.

            Skips over triple-quoted strings to avoid modifying PGML block content.
            """
            result = []
            i = 0
            while i < len(code_str):
                # Skip over triple-quoted strings (don't modify content inside them)
                if i <= len(code_str) - 3 and code_str[i:i+3] == "'''":
                    # Found start of triple-quoted string, copy until end
                    result.append("'''")
                    i += 3
                    # Find the closing '''
                    while i <= len(code_str) - 3:
                        if code_str[i:i+3] == "'''":
                            result.append("'''")
                            i += 3
                            break
                        result.append(code_str[i])
                        i += 1
                    else:
                        # Reached end without finding closing ''', just copy remainder
                        while i < len(code_str):
                            result.append(code_str[i])
                            i += 1
                    continue

                # Also skip double-quoted strings
                if code_str[i] == '"':
                    result.append(code_str[i])
                    i += 1
                    while i < len(code_str):
                        if code_str[i] == '"' and (i == 0 or code_str[i-1] != '\\'):
                            result.append(code_str[i])
                            i += 1
                            break
                        result.append(code_str[i])
                        i += 1
                    continue

                # Also skip single-quoted strings
                if code_str[i] == "'":
                    result.append(code_str[i])
                    i += 1
                    while i < len(code_str):
                        if code_str[i] == "'" and (i == 0 or code_str[i-1] != '\\'):
                            result.append(code_str[i])
                            i += 1
                            break
                        result.append(code_str[i])
                        i += 1
                    continue

                # Look for pattern: word = [
                match = re_module.match(r'(\w+)\s*=\s*\[', code_str[i:])
                if match:
                    # Found a list assignment, find the matching closing bracket
                    var_name = match.group(1)
                    bracket_start = i + match.end() - 1  # Position of the [
                    bracket_count = 0
                    j = bracket_start

                    # Check if this is a PGML answer blank pattern like [_]{...}
                    # If so, skip the PerlList conversion
                    list_content_start = j + 1
                    is_pgml_blank = False

                    # Check if content starts with underscores only: [_+]
                    if list_content_start < len(code_str):
                        temp_j = list_content_start
                        underscore_count = 0
                        while temp_j < len(code_str) and code_str[temp_j] == '_':
                            underscore_count += 1
                            temp_j += 1
                        # If we found only underscores followed by ], this is a PGML blank
                        if underscore_count > 0 and temp_j < len(code_str) and code_str[temp_j] == ']':
                            is_pgml_blank = True

                    if is_pgml_blank:
                        # Skip PerlList conversion for PGML answer blanks
                        # Just copy the original text character by character
                        result.append(code_str[i:j+1])  # Add "var = ["
                        # Find the matching close bracket and keep it as-is
                        bracket_count = 1
                        j += 1
                        while j < len(code_str) and bracket_count > 0:
                            if code_str[j] == '[':
                                bracket_count += 1
                            elif code_str[j] == ']':
                                bracket_count -= 1
                            result.append(code_str[j])
                            j += 1
                        i = j
                    else:
                        # Find matching closing bracket for regular list assignments
                        while j < len(code_str):
                            if code_str[j] == '[':
                                bracket_count += 1
                            elif code_str[j] == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    # Found matching closing bracket
                                    list_literal = code_str[bracket_start:j+1]
                                    result.append(
                                        f'{var_name} = PerlList({list_literal})')
                                    i = j + 1
                                    break
                            j += 1
                        else:
                            # No matching bracket found, keep original
                            result.append(code_str[i])
                            i += 1
                else:
                    result.append(code_str[i])
                    i += 1

            return ''.join(result)

        code = wrap_with_perllist_nested(code)

        # Final cleanup: remove invalid statements like "return = {}"
        # These can appear as artifacts from closure translation
        lines = code.split('\n')
        python_keywords = {'return', 'if', 'else', 'elif',
                           'for', 'while', 'def', 'class', 'import', 'from'}
        cleaned_lines = []
        for line in lines:
            line_stripped = line.strip()
            # Skip lines that try to assign to Python keywords
            if "=" in line_stripped and not line_stripped.startswith("def ") and not line_stripped.startswith("if ") and not line_stripped.startswith("elif ") and not line_stripped.startswith("else") and not line_stripped.startswith("class "):
                var_part = line_stripped.split("=")[0].strip()
                if var_part in python_keywords:
                    # Skip this line - it's invalid
                    continue
            cleaned_lines.append(line)

        code = '\n'.join(cleaned_lines)

        return PreprocessResult(code=code, text_blocks=text_blocks, line_map=line_map)

    def _initialize_arrays(self, code: str) -> str:
        """Initialize arrays/dicts that are assigned to without declaration.

        Detects patterns like:
            x[k] = value
            y[k] = value
            push(answers, ...)

        And adds initialization before the first usage (with less indentation):
            x = {}
            y = {}
            answers = []
            for k in range(...):
                x[k] = value
        """
        import re
        lines = code.split('\n')

        # Track which variables need initialization and where first used
        array_vars = {}  # var_name -> (first_line_num, indentation, is_list)

        # Track whether we're inside a triple-quoted string
        in_triple_quote = False

        # Find all array/dict assignments and push calls
        for line_num, line in enumerate(lines):
            # Toggle triple-quote tracking
            # Count occurrences of ''' on this line
            triple_quote_count = line.count("'''")
            if triple_quote_count > 0:
                # Each pair toggles, odd count means we end in a different state
                in_triple_quote = not in_triple_quote

            # Skip lines that are inside triple-quoted strings
            if in_triple_quote:
                continue

            # Look for patterns like: varname[...] =
            match = re.search(r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*\[', line)
            if match:
                indent = match.group(1)
                var_name = match.group(2)
                if var_name not in array_vars:
                    array_vars[var_name] = (
                        line_num, indent, False)  # False = dict

            # Look for patterns like: push(varname, ...)
            push_match = re.search(
                r'^(\s*)push\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,', line)
            if push_match:
                indent = push_match.group(1)
                var_name = push_match.group(2)
                if var_name not in array_vars:
                    array_vars[var_name] = (
                        line_num, indent, True)  # True = list

        # For each array variable, check if it's already defined and add initialization if not
        insertions = []
        for var_name, (first_use_line, usage_indent, is_list) in sorted(array_vars.items(), reverse=True):
            # Check if variable is already defined before its first use
            already_defined = False
            for i in range(first_use_line):
                # Check for assignments like: var = ... or var[...] = ...
                if re.search(rf'^[^#]*\b{re.escape(var_name)}\s*=', lines[i]):
                    already_defined = True
                    break

            if not already_defined:
                # Find the correct insertion point and indentation
                # We want to insert before the current block starts (before the for/while/if)
                init_indent = ''
                insert_line = first_use_line

                # Go backwards to find the start of the current block
                # The block start is the line with less indentation before the current usage
                found_block_start = False
                for i in range(first_use_line - 1, -1, -1):
                    line = lines[i]
                    # Skip empty lines
                    if not line.strip():
                        continue
                    # Check indentation
                    indent_match = re.match(r'^(\s*)', line)
                    curr_indent = indent_match.group(1) if indent_match else ''

                    # If this line has less indentation than the usage
                    if len(curr_indent) < len(usage_indent):
                        # This is the block statement (for, if, while, etc.)
                        # Insert BEFORE this line, not after
                        init_indent = curr_indent
                        insert_line = i
                        found_block_start = True
                        break

                # If we couldn't find a block statement before (at top level),
                # insert before the first usage
                if not found_block_start:
                    insert_line = first_use_line
                    init_indent = ''

                insertions.append(
                    (insert_line, init_indent, var_name, is_list))

        # Insert initializations in reverse order (to maintain line numbers)
        for line_num, indent, var_name, is_list in sorted(insertions, reverse=True):
            init_val = '[]' if is_list else '{}'
            lines.insert(line_num, f'{indent}{var_name} = {init_val}')

        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Map/Grep block conversion
    # ------------------------------------------------------------------

    def _convert_map_grep_blocks(self, line: str) -> str:
        """Convert Perl map/grep blocks to Python list comprehensions.

        Converts:
            map { EXPR } LIST     ->    [EXPR for _ in LIST]
            grep { EXPR } LIST    ->    [EXPR for _ in LIST if EXPR]
        """
        import re as re_module

        # Match map { ... } expr where expr can be a range like 0 .. 10
        # We need to handle nested braces and capture everything up to the last brace
        def find_map_grep_blocks(text):
            """Find all map/grep blocks in the text and convert them."""
            result = []
            i = 0
            while i < len(text):
                # Look for 'map {' or 'grep {'
                match = re_module.search(r'\b(map|grep)\s*\{', text[i:])
                if not match:
                    result.append(text[i:])
                    break

                # Found map or grep, add everything before it
                result.append(text[i:i + match.start()])
                keyword = match.group(1)
                block_start = i + match.end() - 1  # Position of opening brace

                # Find matching closing brace
                brace_depth = 1
                block_end = block_start + 1
                while block_end < len(text) and brace_depth > 0:
                    if text[block_end] == '{':
                        brace_depth += 1
                    elif text[block_end] == '}':
                        brace_depth -= 1
                    block_end += 1

                if brace_depth == 0:
                    # Extract block content
                    block_content = text[block_start + 1:block_end - 1]

                    # Find the list expression after the closing brace
                    list_start = block_end
                    # Skip whitespace
                    while list_start < len(text) and text[list_start] in ' \t':
                        list_start += 1

                    # Capture list expression (stops at ;, }, or end of common operators)
                    list_end = list_start
                    paren_depth = 0
                    bracket_depth = 0
                    while list_end < len(text):
                        ch = text[list_end]
                        if ch == '(':
                            paren_depth += 1
                        elif ch == ')':
                            paren_depth -= 1
                            if paren_depth < 0:
                                break
                        elif ch == '[':
                            bracket_depth += 1
                        elif ch == ']':
                            bracket_depth -= 1
                            if bracket_depth < 0:
                                break
                        elif ch in ';,}' and paren_depth == 0 and bracket_depth == 0:
                            break
                        elif ch == ' ' and paren_depth == 0 and bracket_depth == 0:
                            # Check if this is the .. range operator
                            if list_end + 3 <= len(text) and text[list_end:list_end+3] == ' ..':
                                list_end += 3
                                while list_end < len(text) and text[list_end] == ' ':
                                    list_end += 1
                                continue
                            else:
                                break
                        list_end += 1

                    list_expr = text[list_start:list_end].strip()

                    # Replace $_ with _ in block
                    block_content = re_module.sub(r'\$_\b', '_', block_content)

                    # Convert fat comma => to =
                    block_content = re_module.sub(
                        r'\s*=>\s*', ' = ', block_content)

                    # Convert Perl range operator .. to Python range()
                    # Handle both "a .. b" and "a..b" formats
                    # First handle parenthesized ranges like (0 .. 3)
                    list_expr = re_module.sub(
                        r'\((\d+)\s*\.\.\s*(\d+)\)', r'range(\1, \2 + 1)', list_expr)
                    # Then handle non-parenthesized ranges
                    list_expr = re_module.sub(
                        r'(\S+)\s*\.\.\s*(\S+)', r'range(\1, \2 + 1)', list_expr)

                    # Build the list comprehension
                    if keyword == 'map':
                        result.append(
                            f"[{block_content} for _ in {list_expr}]")
                    else:  # grep
                        result.append(
                            f"[_ for _ in {list_expr} if {block_content}]")

                    i = list_end
                else:
                    # No matching brace found, just add what we have
                    result.append(text[i:])
                    break

            return ''.join(result)

        return find_map_grep_blocks(line)

    # ------------------------------------------------------------------
    # Grammar and parsing
    # ------------------------------------------------------------------

    def _compile_line(self, line: str) -> List[str]:
        """Compile a single line of PG code into Python lines.

        This method attempts to parse the line with the Lark grammar.  On
        success, it lowers the parse tree into IR and then renders it
        into Python.  If the line cannot be parsed by the grammar, it
        is rewritten using a Pygments based fallback which performs
        conservative token replacement.
        """
        stripped = line.strip()
        indent_prefix = line[: len(line) - len(line.lstrip(' \t'))]
        if not stripped:
            return []
        # Keep comments verbatim
        if stripped.startswith('#'):
            return [stripped]
        # If a Lark parser is available, try to parse.  Fall back to
        # manual parsing and Pygments rewriting on failure.
        if self._parser is not None:
            try:
                tree = self._parser.parse(stripped)
                ir_list = self._transformer.transform(tree)
                if not isinstance(ir_list, list):
                    ir_list = [ir_list]
                result_lines: List[str] = []
                for ir in ir_list:
                    out = self._emit_ir(ir)
                    if out is not None:
                        result_lines.append(out)
                if indent_prefix and result_lines:
                    result_lines = [
                        (indent_prefix + rl) if rl else rl for rl in result_lines
                    ]
                return result_lines if result_lines else []
            except LarkError:
                pass

        # Fallback to Pygments-based token rewriting
        rewritten = self._rewrite_statement(line)
        if not rewritten:
            return []
        return rewritten.split('\n')

    def _compile_expr(self, expr: str) -> str:
        """Compile a small expression using the grammar or fallback rewrite.

        Returns a string containing Python code representing the expression.
        """
        if self._parser is not None:
            try:
                tree = self._parser.parse(expr)
                ir_list = self._transformer.transform(tree)
                # Find first expr or assign
                if not isinstance(ir_list, list):
                    ir_list = [ir_list]
                for ir in ir_list:
                    if ir[0] in ("assign", "expr", "bin", "var", "call"):
                        return self._expr_to_py(ir)
                # Fallback: use pygments rewrite
            except LarkError:
                pass
        return self._rewrite_with_pygments(expr)

    # ------------------------------------------------------------------
    # IR emission
    # ------------------------------------------------------------------

    def _desigil(self, name: str) -> str:
        """Remove leading sigil characters from Perl variable names."""
        return self._pygments_rewriter.desigil(name)

    def _rewrite_with_pygments(self, code: str) -> str:
        """Fallback rewrite using Pygments for conservative token replacement."""
        return self._pygments_rewriter.rewrite_with_pygments(code)

    # ------------------------------------------------------------------
    # IR Emission Delegation
    # ------------------------------------------------------------------

    def _emit_ir(self, ir: Any, indent: int = 0) -> Optional[str]:
        """Delegate to IR emitter."""
        if self._ir_emitter:
            return self._ir_emitter.emit_ir(ir, indent)
        return None

    def _emit_block(self, block_ir: Any, indent: int) -> List[str]:
        """Delegate to IR emitter."""
        if self._ir_emitter:
            return self._ir_emitter.emit_block(block_ir, indent)
        return []

    def _emit_closure(self, body_stmts: List[Any], indent: int, context_name: str = "func") -> Any:
        """Delegate to IR emitter."""
        if self._ir_emitter:
            return self._ir_emitter.emit_closure(body_stmts, indent, context_name)
        return None

    def _expr_to_py(self, expr: Any) -> str:
        """Delegate to IR emitter."""
        if self._ir_emitter:
            return self._ir_emitter.expr_to_py(expr)
        return str(expr)

    # ------------------------------------------------------------------
    # Structured rewriting helpers
    # ------------------------------------------------------------------

    def _try_rewrite_for_loop(
        self, lines: List[str], start_index: int
    ) -> tuple[List[str], int] | None:
        """Rewrite simple Perl for/foreach loops into Python for loops."""
        line = lines[start_index]
        # Match two patterns:
        # 1. for VAR (expr) { ... }       -> VAR is captured
        # 2. for (expr) { ... }           -> VAR is implicit $_
        for_match = re.match(
            r"(\s*)(?:for|foreach)\s+(?:my\s+)?((?:[$@%]?[A-Za-z_][\w]*)?)\s*\(([^)]*)\)\s*\{",
            line,
        )
        if not for_match:
            return None

        indent = for_match.group(1)
        iterator_token = for_match.group(
            2).strip() if for_match.group(2) else None
        iterable_expr = for_match.group(3).strip()

        # If no iterator token, default to $_
        if not iterator_token:
            iterator_token = '$_'

        block_lines: List[str] = [line]
        brace_depth = line.count('{') - line.count('}')
        idx = start_index + 1
        while idx < len(lines) and brace_depth > 0:
            current_line = lines[idx]
            block_lines.append(current_line)
            brace_depth += current_line.count('{') - current_line.count('}')
            idx += 1

        if brace_depth != 0:
            return None

        loop_var = self._desigil(iterator_token)
        iterable_py = self._convert_for_iterable(iterable_expr)
        if not iterable_py:
            return None

        body_lines: List[str] = []
        tail_lines: List[str] = []

        open_index = line.find('{')
        if open_index == -1:
            return None

        body_candidates: List[str] = [line[open_index + 1:]]
        body_candidates.extend(block_lines[1:])

        for idx, candidate in enumerate(body_candidates):
            if candidate is None:
                continue
            candidate_text = candidate.rstrip('\n')
            is_last = idx == len(body_candidates) - 1
            if is_last:
                closing_index = candidate_text.rfind('}')
                if closing_index != -1:
                    body_part = candidate_text[:closing_index].rstrip()
                    if body_part.strip():
                        body_lines.append(body_part)
                    tail = candidate_text[closing_index + 1:].strip()
                    if tail:
                        tail_lines.append(f"{indent}{tail}")
                else:
                    if candidate_text.strip():
                        body_lines.append(candidate_text)
            else:
                if candidate_text.strip():
                    body_lines.append(candidate_text)

        compiled_body_lines: List[str] = []
        body_indent = indent + '    '

        # Process body lines, handling PGML blocks specially
        idx = 0
        while idx < len(body_lines):
            body_line = body_lines[idx]

            # Check if this line starts a PGML block
            if re.search(r'BEGIN_PGML\s*$', body_line):
                # Collect PGML block content
                pgml_content_lines: List[str] = []
                idx += 1
                while idx < len(body_lines) and not re.match(r'END_PGML', body_lines[idx]):
                    pgml_content_lines.append(body_lines[idx])
                    idx += 1

                # Transform the PGML evaluators within the block
                pgml_content = "\n".join(pgml_content_lines)
                transformed_pgml = self._transform_pgml_evaluators(
                    pgml_content)
                escaped_content = self._escape_triple_quotes(transformed_pgml)

                # Create PGML block variable
                block_var = f"PGML_BLOCK_LOOP_{idx}"
                compiled_body_lines.append(
                    f"{body_indent}{block_var} = '''\n{escaped_content}\n'''")
                compiled_body_lines.append(
                    f"{body_indent}TEXT(PGML({block_var}))")
                idx += 1
                continue

            # Check if this line is a method-call style BEGIN_TIKZ or BEGIN_LATEX_IMAGE
            # Also handle array indexing like $graph[$i]->BEGIN_TIKZ
            tikz_match = re.search(
                r'(\$\w+(?:\[[^\]]*\])*)\s*->\s*BEGIN_TIKZ', body_line)
            latex_match = re.search(
                r'(\$\w+(?:\[[^\]]*\])*)\s*->\s*BEGIN_LATEX_IMAGE', body_line)

            if (tikz_match or latex_match) and body_line.strip().endswith(('BEGIN_TIKZ', 'BEGIN_LATEX_IMAGE')):
                obj_var = (tikz_match or latex_match).group(1)
                end_marker = "END_TIKZ" if tikz_match else "END_LATEX_IMAGE"
                method_name = "BEGIN_TIKZ" if tikz_match else "BEGIN_LATEX_IMAGE"

                # Collect content until END marker
                content_lines: List[str] = []
                idx += 1
                while idx < len(body_lines) and not re.match(rf'^\s*{end_marker}\s*$', body_lines[idx]):
                    content_lines.append(body_lines[idx])
                    idx += 1

                # Join content and escape for raw string
                content = '\n'.join(content_lines)
                escaped_content = content.replace("'''", r"\'\'\'")

                # Convert $obj_var to Python name
                py_var = obj_var[1:] if obj_var.startswith('$') else obj_var

                # Emit the method call with content as raw string argument
                # Split into multiple lines to avoid embedding newlines in f-string
                compiled_body_lines.append(
                    f"{body_indent}{py_var}.{method_name}(r'''")
                # Add content lines with proper indentation
                for content_line in content_lines:
                    compiled_body_lines.append(content_line)
                compiled_body_lines.append(f"{body_indent}''')")
                idx += 1  # Skip the END marker
                continue

            # Regular line - compile it
            compiled = self._compile_line(body_line)
            for compiled_line in compiled:
                stripped_total = compiled_line.strip()
                if not stripped_total:
                    continue
                inner_indent, inner_body = self._split_indent(compiled_line)
                inner_body = re.sub(r'^my\s+', '', inner_body)
                compiled_body_lines.append(
                    f"{body_indent}{inner_indent}{inner_body}")

            idx += 1

        if not compiled_body_lines:
            compiled_body_lines.append(f"{body_indent}pass")

        rewritten_loop = [f"{indent}for {loop_var} in {iterable_py}:"]
        rewritten_loop.extend(compiled_body_lines)

        for tail_line in tail_lines:
            tail_compiled = self._compile_line(tail_line)
            rewritten_loop.extend(tail_compiled)

        return rewritten_loop, len(block_lines)

    def _convert_for_iterable(self, iterable_expr: str) -> str | None:
        expr = iterable_expr.strip()
        if not expr:
            return None

        if '..' in expr:
            start_raw, end_raw = expr.split('..', 1)
            start_py = self._convert_range_bound(start_raw.strip())
            end_py = self._convert_range_bound(end_raw.strip())
            if start_py is None or end_py is None:
                return None
            return f"range({start_py}, ({end_py}) + 1)"

        if expr.startswith('$#'):
            array_name = expr[2:].strip()
            if array_name.startswith('{') and array_name.endswith('}'):
                array_name = array_name[1:-1].strip()
            array_py = self._desigil(f'@{array_name}')
            return f"len({array_py}) - 1"

        compiled = self._compile_expr(expr).strip()
        if compiled.startswith('(') and compiled.endswith(')'):
            compiled = compiled[1:-1]
        return compiled

    def _convert_range_bound(self, expr: str) -> str | None:
        expr = expr.strip()
        if not expr:
            return '0'

        if expr.startswith('$#'):
            array_name = expr[2:].strip()
            if array_name.startswith('{') and array_name.endswith('}'):
                array_name = array_name[1:-1].strip()
            array_py = self._desigil(f'@{array_name}')
            return f"len({array_py}) - 1"

        if expr.startswith('$') or expr.startswith('@'):
            return self._desigil(expr)

        compiled = self._compile_expr(expr).strip()
        if compiled.startswith('(') and compiled.endswith(')'):
            compiled = compiled[1:-1]
        return compiled

    def _rewrite_statement(self, line: str) -> str:
        """Rewrite a single Perl-like statement using Pygments and helpers."""
        indent, body = self._split_indent(line)
        stripped = body.strip()
        if not stripped:
            return ''
        if stripped.startswith('#'):
            return line
        if 'loadMacros' in stripped:
            cleaned = self._strip_load_macros(stripped)
            if not cleaned:
                return ''
            stripped = cleaned
        if stripped == '}':
            return ''
        if re.fullmatch(r'\}\s*else\s*\{', stripped):
            return f'{indent}else:'

        control = self._rewrite_control_flow(indent, stripped)
        if control is not None:
            return control

        rewritten = self._rewrite_with_pygments(body)
        # Apply string interpolation (still needed for token-level $var → f-string)
        rewritten = self._convert_string_interpolation(rewritten)
        return indent + rewritten if rewritten else ''

    def _split_indent(self, line: str) -> tuple[str, str]:
        match = re.match(r'(\s*)(.*)', line)
        if not match:
            return '', line
        return match.group(1), match.group(2)

    def _strip_load_macros(self, line: str) -> str:
        if 'loadMacros' not in line:
            return line
        parts = [part for part in line.split(';') if 'loadMacros' not in part]
        return '; '.join(part.strip() for part in parts if part.strip())

    def _rewrite_control_flow(self, indent: str, stripped: str) -> str | None:
        header_keywords = ('if', 'elsif', 'unless', 'while')
        for keyword in header_keywords:
            if stripped.startswith(keyword):
                parsed = self._parse_conditional_header(stripped, keyword)
                if not parsed:
                    break
                rest, condition, tail = parsed
                cond_py = self._compile_expr(condition)
                # Strip outer parens if already present (binary exprs add them)
                if cond_py.startswith('(') and cond_py.endswith(')'):
                    cond_py = cond_py[1:-1]
                py_keyword = keyword
                if keyword == 'elsif':
                    py_keyword = 'elif'
                elif keyword == 'unless':
                    py_keyword = 'if not'
                header = f'{indent}{py_keyword} ({cond_py}):'
                tail = tail.strip()
                if tail:
                    rewritten_tail = self._rewrite_statement(
                        f'{indent}    {tail}')
                    if rewritten_tail:
                        return f"{header}\n{rewritten_tail}"
                return header
        return None

    def _parse_conditional_header(self, stripped: str, keyword: str) -> tuple[str, str, str] | None:
        prefix_len = len(keyword)
        remainder = stripped[prefix_len:].lstrip()
        if not remainder.startswith('('):
            return None
        depth = 0
        condition_chars: List[str] = []
        tail_start = None
        for idx, ch in enumerate(remainder):
            if ch == '(':
                depth += 1
                if depth == 1:
                    continue
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    tail_start = idx + 1
                    break
            if depth >= 1:
                condition_chars.append(ch)
        if tail_start is None:
            return None
        condition = ''.join(condition_chars).strip()
        tail = remainder[tail_start:].lstrip()
        if tail.startswith('{'):
            tail = tail[1:].lstrip()
        if tail.endswith('}'):
            tail = tail[:-1].rstrip()
        return keyword, condition, tail

    def _convert_string_interpolation(self, line: str) -> str:
        def repl(match: re.Match[str]) -> str:
            quote = match.group(1)
            content = match.group(2)
            if quote == '"' and '$' in content:
                escaped = content.replace('{', '{{').replace('}', '}}')
                converted = re.sub(
                    r'\$([a-zA-Z_][a-zA-Z0-9_]*)', r'{\1}', escaped)
                return f'f"{converted}"'
            return match.group(0)

        return re.sub(r'(["\'])((?:[^\\]|\\.)*?)\1', repl, line)

    # ------------------------------------------------------------------
    # Text block processing
    # ------------------------------------------------------------------

    def _escape_triple_quotes(self, text: str) -> str:
        """Escape special characters in text for Python triple quoted string literals."""
        return self._block_extractor.escape_triple_quotes(text)

    def _strip_inline_comment(self, line: str) -> str:
        """
        Remove inline Perl/Python comments from a line.

        This is needed when joining multi-line statements to prevent comments
        from eating subsequent code. For example:
            func(arg1,    # comment
                 arg2)
        Should become:
            func(arg1, arg2)
        Not:
            func(arg1,    # comment arg2)

        Args:
            line: Line potentially containing # comment

        Returns:
            Line with inline comment removed, trailing whitespace stripped
        """
        # Find # that's not inside a string
        in_string = False
        string_char = None
        escaped = False

        for i, char in enumerate(line):
            if escaped:
                escaped = False
                continue

            if char == '\\':
                escaped = True
                continue

            if char in ('"', "'"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None

            elif char == '#' and not in_string:
                # Check if this is a comment or the $# operator
                # $# is a Perl operator for array last index, not a comment
                if i > 0 and line[i-1] == '$':
                    # This is $#, not a comment - continue processing
                    continue
                # Found comment start - return everything before it
                return line[:i].rstrip()

        # No comment found
        return line.rstrip()

    def _transform_load_macros(self, macro_list_str: str) -> Tuple[List[str], str]:
        """
        Transform loadMacros() call to Python imports using the macro registry.

        Generates comprehensive imports based on the macro registry which maps
        Perl macro names to Python modules and their exported functions.
        """
        import re

        # Import registry from pg.macros package
        try:
            from pg.macros.registry import get_macro_info
        except ImportError:
            # Fallback if registry not available
            return [], "# loadMacros() - registry not available"

        # Parse macro names from the loadMacros call
        macros = re.findall(r'"([^"]+)"|\'([^\']+)\'', macro_list_str)
        flattened = [a or b for a, b in macros]

        # Collect imports by module to deduplicate
        imports_by_module: dict[str, set[str]] = {}
        # Modules to import without 'from'
        module_level_imports: list[str] = []
        loaded_macros: list[str] = []
        skipped_macros: list[str] = []

        for macro in flattened:
            info = get_macro_info(macro)
            if info and info.get("module"):
                module = info["module"]
                functions = info.get("functions", [])

                if functions:
                    # Add specific function imports (from X import Y)
                    if module not in imports_by_module:
                        imports_by_module[module] = set()
                    imports_by_module[module].update(functions)
                    loaded_macros.append(macro)
                elif module:
                    # Module exists but no functions listed - import entire module
                    if module not in module_level_imports:
                        module_level_imports.append(module)
                    loaded_macros.append(macro)
                else:
                    skipped_macros.append(macro)
            else:
                skipped_macros.append(macro)

        # Generate import statements
        import_lines = []

        # For module-level imports (empty function lists), use 'from X import *'
        # This makes all functions directly accessible without needing module prefix
        if module_level_imports:
            for module in sorted(module_level_imports):
                import_lines.append(f"from {module} import *")

        # Then, generate function-specific imports: from X import Y, Z
        for module in sorted(imports_by_module.keys()):
            functions = imports_by_module[module]
            func_list = ", ".join(sorted(functions))
            import_lines.append(f"from {module} import {func_list}")

        # Generate comment
        comment_parts = []
        if loaded_macros:
            comment_parts.append(f"# Loaded: {', '.join(loaded_macros)}")
        if skipped_macros:
            comment_parts.append(
                f"# Skipped (not in registry): {', '.join(skipped_macros)}")
        comment = " | ".join(
            comment_parts) if comment_parts else "# loadMacros() - no macros"

        return import_lines, comment


def convert_pg_file(
    source_path: str | Path,
    *,
    output_path: str | Path | None = None,
    use_sandbox_macros: bool = False,
    overwrite: bool = False,
    encoding: str = "utf-8",
    preprocessor: PGPreprocessor | None = None,
    standalone: bool = False,
) -> tuple[Path, PreprocessResult]:
    """Convert a .pg file to Python using the Pygments/Lark preprocessor."""

    pg_path = Path(source_path)
    if not pg_path.exists():
        raise FileNotFoundError(f"PG source file not found: {pg_path}")

    processor = preprocessor or PGPreprocessor()
    pg_source = pg_path.read_text(encoding=encoding)
    result = processor.preprocess(
        pg_source, use_sandbox_macros=use_sandbox_macros, standalone=standalone)

    # Use .py extension for standalone files, .pyg for sandboxed files
    default_suffix = '.py' if standalone else '.pyg'
    output = Path(output_path) if output_path else pg_path.with_suffix(
        default_suffix)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.code, encoding=encoding)
    return output, result
