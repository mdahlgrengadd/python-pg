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

# Import Pygments for token aware rewriting
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
        # Initialise Pygments lexer once
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
                self._grammar(),
                start="start",
                parser="earley",
                maybe_placeholders=True,
                propagate_positions=True,
            )
            self._transformer = self._make_transformer()
        else:
            self._parser = None
            self._transformer = None

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
        pg_source = self._convert_heredocs_global(pg_source)

        # Fix reference dereference arrows before parsing
        # $x->[$i] -> $x[i], $x->{key} -> $x[key]
        pg_source = self._fix_reference_dereferences(pg_source)

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
                        stripped_no_comment = self._strip_inline_comment(
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
                            check_line = self._strip_inline_comment(stripped)
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
                            line_without_comment = self._strip_inline_comment(
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
                        transformed_pgml = self._transform_pgml_evaluators(
                            block_content)
                        block_var = f"PGML_BLOCK_{len(text_blocks) - 1}"
                        escaped_content = self._escape_triple_quotes(
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
                        transformed_content = self._transform_text_block(
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

    def _grammar(self) -> str:
        """Return the Lark grammar for parsing PG/Perl statements.

        This extended grammar covers most Perl constructs found in PG files:
        - Control flow: if/elsif/unless/while/for
        - Ternary operator: cond ? true : false
        - Hash/array access: $hash{key}, $array[idx]
        - Method calls: $obj->method()
        - Ranges: 0..10
        - Map/grep blocks
        - Statement modifiers: stmt if cond
        - Fat comma: key => value

        The grammar is designed to avoid reduce/reduce conflicts.
        Unparseable constructs fall back to Pygments rewriting.
        """
        return r"""
            start: (stmt ";"?)*

            stmt: if_stmt
                | while_stmt
                | for_stmt
                | foreach_stmt
                | do_until_stmt
                | decl
                | assign
                | return_stmt
                | expr_stmt
                | document
                | enddocument
                | stmt_modifier

            // Control flow statements
            if_stmt: "if" "(" expr ")" block elsif_clause* else_clause?
            elsif_clause: "elsif" "(" expr ")" block
            else_clause: "else" block
            unless_stmt: "unless" "(" expr ")" block
            while_stmt: "while" "(" expr ")" block
            for_stmt: "for" "my"? var "(" expr ")" block
            foreach_stmt: "foreach" "my"? var "(" expr ")" block
            do_until_stmt: "do" block "until" "("? expr ")"?

            block: "{" (stmt ";"?)* "}"

            // Statement modifiers (trailing conditionals)
            stmt_modifier_if: simple_stmt "if" expr        -> stmt_modifier_if
            stmt_modifier_unless: simple_stmt "unless" expr -> stmt_modifier_unless
            stmt_modifier: stmt_modifier_if | stmt_modifier_unless
            simple_stmt: decl | assign | call_expr

            document: "DOCUMENT" "(" ")"            -> document_call
            enddocument: "ENDDOCUMENT" "(" ")"      -> enddocument_call
            loadmacros: "loadMacros" "(" /[^)]*/ ")" -> loadmacros_call

            decl: "my" var "=" expr               -> assign_stmt
                | "my" var "=" sub_closure      -> assign_stmt
            assign: var "=" expr                -> assign_stmt
                  | var "=" sub_closure        -> assign_stmt
                  | var subscript "=" expr      -> subscript_assign
            expr_stmt: expr                      -> expr_stmt

            // Sub closures: sub { ... }
            sub_closure: "sub" "{" closure_body "}"
            closure_body: (closure_stmt (";" | "\n")?)*
            closure_stmt: param_decl
                        | decl
                        | assign
                        | if_stmt
                        | while_stmt
                        | for_stmt
                        | foreach_stmt
                        | do_until_stmt
                        | return_stmt
                        | expr_stmt
                        | stmt_modifier

            // Parameter unpacking: my ($a, $b) = @_;  or my ($a, $b) = @$arr;
            param_decl: "my" "(" param_var_list ")" "=" array_deref_special
            param_var_list: var ("," var)*
            array_deref_special: "@" "_"  -> deref_args
                               | "@" "$" NAME  -> deref_var

            // Return statement: return expr; or return [expr, ...];
            // Must handle [ ... ] specially because [ is ambiguous (could be subscript or array literal)
            return_stmt: "return" return_expr_special
            return_expr_special: "[" return_list "]"  -> return_array
                               | "[" "]"              -> return_empty_array
                               | expr
            return_list: expr ("," expr)*

            // Args can be comma-separated expressions or named parameters with =>
            args: arg_item ("," arg_item)*
            arg_item: expr ("=>" expr)?

            // Hash/Array subscripting
            subscript: "[" expr "]"              -> array_subscript
                     | "{" expr "}"              -> hash_subscript

            // Expressions with precedence (lowest to highest)
            ?expr: ternary_expr

            // Ternary: cond ? true : false
            ?ternary_expr: or_expr ("?" or_expr ":" ternary_expr)?  -> ternary_expr

            ?or_expr: and_expr ((OR_OP | "or") and_expr)*      -> binary_expr
            ?and_expr: comp_expr ((AND_OP | "and") comp_expr)*  -> binary_expr

            // Comparison operators (eq, ne, lt, gt, le, ge, ==, !=, <, >, <=, >=)
            ?comp_expr: range_expr (comp_op range_expr)*     -> binary_expr
            comp_op: EQ | NE | LT | GT | LE | GE | EQEQ | BANGEQ | LANGLE | RANGLE | LTEQ | GTEQ
            EQ: "eq"
            NE: "ne"
            LT: "lt"
            GT: "gt"
            LE: "le"
            GE: "ge"
            EQEQ: "=="
            BANGEQ: "!="
            LANGLE: "<"
            RANGLE: ">"
            LTEQ: "<="
            GTEQ: ">="
            
            // Logical operators (must be terminals, not literal strings, to avoid Lark confusion)
            OR_OP: "||"
            AND_OP: "&&"

            // Range operator: 0..10
            ?range_expr: add_expr (".." add_expr)?           -> range_expr

            ?add_expr: mul_expr (add_op mul_expr)*           -> binary_expr
            add_op: PLUS | MINUS | DOT
            PLUS.2: "+"
            MINUS.2: "-"
            DOT.2: "."

            ?mul_expr: unary_expr (mul_op unary_expr)*       -> binary_expr
            mul_op: STAR | SLASH | PERCENT | X
            STAR.2: "*"
            SLASH.2: "/"
            PERCENT.2: "%"
            X.2: "x"

            ?unary_expr: postfix_expr
                       | "-" unary_expr                      -> unary_minus
                       | "!" unary_expr                      -> unary_not

            // Postfix: method calls, subscripts
            ?postfix_expr: primary (postfix_op)*
            postfix_op: "->" NAME "(" args? ")"              -> method_call
                      | subscript

            ?primary: call_expr | var | atom | array_deref | "(" expr ")"

            call_expr: NAME "(" args? ")"          -> call_expr

            // Map and grep blocks
            map_expr: "map" "{" expr "}" expr                -> map_expr
            grep_expr: "grep" "{" expr "}" expr              -> grep_expr

            // Array dereferencing: @$var or @_
            array_deref: "@" "$" NAME                        -> array_deref_var
                       | "@" "_"                             -> array_deref_args

            var: VAR
            atom: NUMBER | STRING | NAME | regex_literal

            regex_literal: QR "/" /[^\/]+/ "/" REGEX_FLAGS?  -> regex_literal

            QR.2: "qr"
            NAME: /[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*/  
            VAR: /[\$@%][A-Za-z_][A-Za-z0-9_]*/
            STRING: /"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'/
            NUMBER: /[0-9]+(?:\.[0-9]+)?/
            REGEX_FLAGS: /[imsxo]+/

            %import common.WS
            %ignore WS
            """

    def _make_transformer(self) -> Transformer:
        """Create and return a Lark transformer that lowers parse trees."""

        @v_args(inline=True)
        class ToIR(Transformer):
            """Lower the parse tree into intermediate representation (IR)."""

            def start(self, *stmts):
                """Flatten the list of statements from the top-level start rule."""
                return list(stmts)

            def stmt(self, item):
                """Unwrap the single child of a stmt production."""
                return item

            # Document control
            def document_call(self):
                return ("call", "DOCUMENT", [])

            def enddocument_call(self):
                return ("call", "ENDDOCUMENT", [])

            def loadmacros_call(self, *args):
                return ("noop", )

            # Control flow statements
            def if_stmt(self, condition, block, *clauses):
                """Lower if statement with optional elsif/else clauses."""
                return ("if", condition, block, list(clauses))

            def elsif_clause(self, condition, block):
                return ("elsif", condition, block)

            def else_clause(self, block):
                return ("else", block)

            def unless_stmt(self, condition, block):
                """Lower unless statement (if not)."""
                return ("unless", condition, block)

            def while_stmt(self, condition, block):
                """Lower while loop."""
                return ("while", condition, block)

            def for_stmt(self, var, expr, block):
                """Lower for loop."""
                return ("for", var, expr, block)

            def foreach_stmt(self, var, expr, block):
                """Lower foreach loop."""
                return ("foreach", var, expr, block)

            def do_until_stmt(self, block, condition):
                """Lower do-until loop."""
                return ("do_until", block, condition)

            def block(self, *stmts):
                """Lower block of statements."""
                return ("block", list(stmts))

            # Statement modifiers
            def stmt_modifier(self, child):
                """Pass through stmt_modifier_if or stmt_modifier_unless."""
                return child

            def stmt_modifier_if(self, stmt, condition):
                """Lower statement modifier with 'if'."""
                return ("stmt_modifier", stmt, "if", condition)

            def stmt_modifier_unless(self, stmt, condition):
                """Lower statement modifier with 'unless'."""
                return ("stmt_modifier", stmt, "unless", condition)

            def simple_stmt(self, stmt):
                return stmt

            # Assignments
            def assign_stmt(self, var, expr):
                """Lower a variable declaration or assignment."""
                return ("assign", var, expr)

            def subscript_assign(self, var, subscript, expr):
                """Lower subscript assignment: $arr[0] = value."""
                return ("subscript_assign", var, subscript, expr)

            # Subscripting
            def array_subscript(self, expr):
                return ("array_subscript", expr)

            def hash_subscript(self, expr):
                return ("hash_subscript", expr)

            # Expressions
            def call_expr(self, name, *args):
                """Lower a function call expression."""
                arglist = args[0] if args else []
                # Convert Perl namespace operator :: to Python dot notation
                name = name.replace("::", ".") if isinstance(
                    name, str) else name
                return ("call", name, arglist)

            def expr_stmt(self, expr):
                """Lower an expression statement."""
                return ("expr", expr)

            def ternary_expr(self, *parts):
                """Lower ternary operator: cond ? true : false."""
                if len(parts) == 1:
                    return parts[0]
                elif len(parts) == 3:
                    cond, true_val, false_val = parts
                    return ("ternary", cond, true_val, false_val)
                return parts[0]

            def binary_expr(self, left, *rest):
                """Lower binary operations."""
                expr = left
                for op, right in zip(rest[::2], rest[1::2]):
                    # Extract operator string from Tree or Token
                    if hasattr(op, 'children') and op.children:
                        # If op is a Tree with children, get the first child
                        op_tok = op.children[0]
                    elif hasattr(op, 'data'):
                        # If op is a Tree without children (inline rules), use a Token
                        # This shouldn't happen with properly defined grammars
                        op_tok = op
                    else:
                        # op is already a Token or string
                        op_tok = op

                    # Extract the actual operator string value
                    if hasattr(op_tok, 'value'):
                        op_str = op_tok.value
                    elif hasattr(op_tok, 'type'):
                        # Token without value attribute
                        op_str = str(op_tok)
                    else:
                        op_str = str(op_tok)

                    expr = ("bin", expr, op_str, right)
                return expr

            # Operator extractors - these are needed because operators are defined as rules
            def add_op(self, token):
                """Extract add operator token."""
                # token is now a Token object from the terminal
                if hasattr(token, 'value'):
                    return token.value
                else:
                    return str(token)

            def mul_op(self, token):
                """Extract mul operator token."""
                # token is now a Token object from the terminal
                if hasattr(token, 'value'):
                    return token.value
                else:
                    return str(token)

            def comp_op(self, token):
                """Extract comparison operator token."""
                # Tokens like EQ, GT, etc. come through as Token objects
                if hasattr(token, 'type'):
                    # Map token types to operator strings
                    op_map = {
                        'EQ': 'eq', 'NE': 'ne', 'LT': 'lt', 'GT': 'gt',
                        'LE': 'le', 'GE': 'ge', 'EQEQ': '==', 'BANGEQ': '!=',
                        'LANGLE': '<', 'RANGLE': '>', 'LTEQ': '<=', 'GTEQ': '>='
                    }
                    return op_map.get(token.type, token.value)
                if hasattr(token, 'value'):
                    return token.value
                return str(token)

            def range_expr(self, *parts):
                """Lower range operator: 0..10."""
                if len(parts) == 2:
                    start, end = parts
                    return ("range", start, end)
                return parts[0]

            def unary_minus(self, expr):
                return ("unary", "-", expr)

            def unary_not(self, expr):
                return ("unary", "!", expr)

            def postfix_expr(self, primary, *postfix_ops):
                """Lower postfix operations (method calls, subscripts)."""
                expr = primary
                for op in postfix_ops:
                    expr = ("postfix", expr, op)
                return expr

            def postfix_op(self, child):
                """Pass through the postfix operation (method_call or subscript)."""
                return child

            def method_call(self, name, *args):
                """Lower method call: ->method()."""
                arglist = args[0] if args else []
                return ("method_call", name, arglist)

            # Map and grep
            def map_expr(self, block_expr, list_expr):
                """Lower map block."""
                return ("map", block_expr, list_expr)

            def grep_expr(self, block_expr, list_expr):
                """Lower grep block."""
                return ("grep", block_expr, list_expr)

            # Regex
            def regex_literal(self, *args):
                """Lower regex literal: qr/pattern/flags."""
                # Can receive (qr_token, pattern, flags) or (pattern, flags) depending on parsing
                if len(args) == 3:
                    qr, pattern, flags = args
                elif len(args) == 2:
                    pattern, flags = args
                else:
                    # Fallback
                    pattern = args[0] if args else ""
                    flags = ""
                return ("regex", str(pattern), str(flags))

            # Tokens
            def VAR(self, tok):
                return ("var", str(tok))

            def NAME(self, tok):
                return str(tok)

            def NUMBER(self, tok):
                return str(tok)

            def STRING(self, tok):
                return str(tok)

            def args(self, *items):
                return list(items)

            def arg_item(self, *children):
                """Handle argument item: expr or expr => expr"""
                if len(children) == 1:
                    # Just an expression
                    return children[0]
                elif len(children) == 2:
                    # expr => expr (named parameter)
                    key_expr, val_expr = children
                    return ("named_param", key_expr, val_expr)
                else:
                    return children[0]

            def var(self, child):
                """Unwrap the var rule to return its child."""
                return child

            def atom(self, child):
                """Unwrap the atom rule to return its child."""
                return child

            # Sub closures
            def sub_closure(self, body):
                """Lower sub { ... } closure."""
                return ("closure", body)

            def closure_body(self, *stmts):
                """Lower closure body statements."""
                return list(stmts)

            def closure_stmt(self, stmt):
                """Unwrap closure statement."""
                return stmt

            # Parameter unpacking
            def param_decl(self, var_list, deref):
                """Lower my ($a, $b) = @_; or my ($a, $b) = @$arr;"""
                return ("param_unpack", var_list, deref)

            def param_var_list(self, *vars):
                """Lower parameter variable list."""
                return list(vars)

            # Array dereferencing in parameter context
            def deref_args(self):
                """Lower @_ dereference."""
                return ("special_var", "@_")

            def deref_var(self, name):
                """Lower @$var dereference."""
                return ("array_deref", ("var", f"${name}"))

            # Return statements
            def return_stmt(self, value):
                """Lower return statement."""
                return ("return", value)

            def return_expr_special(self, expr):
                """Unwrap return expression special (handles both arrays and regular exprs)."""
                return expr

            def return_array(self, expr_list):
                """Lower return [...];"""
                # expr_list is the result of return_list which is a list of expressions
                return ("array", expr_list if isinstance(expr_list, list) else [expr_list])

            def return_expr(self, expr):
                """Lower return expr;"""
                return expr

            def return_list(self, *exprs):
                """Lower comma-separated return list."""
                return list(exprs)

            def return_empty_array(self):
                """Lower return [] (empty array)."""
                return ("array", [])

            # Array dereferencing in expressions
            def array_deref_var(self, name):
                """Lower @$name array dereference."""
                return ("array_deref", ("var", f"${name}"))

            def array_deref_args(self):
                """Lower @_ array dereference."""
                return ("special_var", "@_")

        return ToIR()

    # ------------------------------------------------------------------
    # Compilation helpers
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

    def _emit_ir(self, ir: Any, indent: int = 0) -> Optional[str]:
        """Convert IR nodes into Python code lines. Returns None for no output."""
        typ = ir[0]
        ind = "    " * indent

        if typ == "noop":
            return None

        # Control flow statements
        if typ == "if":
            _, condition, block, clauses = ir
            cond_py = self._expr_to_py(condition)
            block_stmts = self._emit_block(block, indent + 1)
            lines = [f"{ind}if {cond_py}:"]
            lines.extend(block_stmts)

            # Handle elsif and else clauses
            for clause in clauses:
                if clause[0] == "elsif":
                    _, elif_cond, elif_block = clause
                    elif_cond_py = self._expr_to_py(elif_cond)
                    lines.append(f"{ind}elif {elif_cond_py}:")
                    lines.extend(self._emit_block(elif_block, indent + 1))
                elif clause[0] == "else":
                    _, else_block = clause
                    lines.append(f"{ind}else:")
                    lines.extend(self._emit_block(else_block, indent + 1))

            return "\n".join(lines)

        if typ == "unless":
            _, condition, block = ir
            cond_py = self._expr_to_py(condition)
            block_stmts = self._emit_block(block, indent + 1)
            lines = [f"{ind}if not ({cond_py}):"]
            lines.extend(block_stmts)
            return "\n".join(lines)

        if typ == "while":
            _, condition, block = ir
            cond_py = self._expr_to_py(condition)
            block_stmts = self._emit_block(block, indent + 1)
            lines = [f"{ind}while {cond_py}:"]
            lines.extend(block_stmts)
            return "\n".join(lines)

        if typ in {"for", "foreach"}:
            _, var, expr, block = ir
            var_name = self._desigil(var[1] if isinstance(var, tuple) else var)
            expr_py = self._expr_to_py(expr)
            block_stmts = self._emit_block(block, indent + 1)
            lines = [f"{ind}for {var_name} in {expr_py}:"]
            lines.extend(block_stmts)
            return "\n".join(lines)

        if typ == "do_until":
            _, block, condition = ir
            cond_py = self._expr_to_py(condition)
            block_stmts = self._emit_block(block, indent + 1)
            lines = [f"{ind}while True:"]
            lines.extend(block_stmts)
            lines.append(f"{ind}    if ({cond_py}):")
            lines.append(f"{ind}        break")
            return "\n".join(lines)

        if typ == "stmt_modifier":
            _, stmt, modifier, condition = ir
            stmt_py = self._emit_ir(stmt, indent)
            cond_py = self._expr_to_py(condition)
            # Strip outer parens if already present (binary exprs add them)
            if cond_py.startswith('(') and cond_py.endswith(')'):
                cond_py = cond_py[1:-1]
            if modifier == "if":
                return f"{ind}if ({cond_py}): {stmt_py.strip()}"
            else:  # unless
                return f"{ind}if not ({cond_py}): {stmt_py.strip()}"

        # Assignments
        if typ == "assign":
            _, var, expr = ir
            var_name = self._desigil(var[1] if isinstance(var, tuple) else var)
            expr_py = self._expr_to_py(expr)
            # Strip outer parens from binary exprs in assignments (cleaner output)
            if expr_py.startswith('(') and expr_py.endswith(')') and isinstance(expr, tuple) and expr[0] == "bin":
                expr_py = expr_py[1:-1]
            return f"{ind}{var_name} = {expr_py}"

        if typ == "subscript_assign":
            _, var, subscript, expr = ir
            var_name = self._desigil(var[1] if isinstance(var, tuple) else var)
            subscript_py = self._emit_subscript(subscript)
            expr_py = self._expr_to_py(expr)
            return f"{ind}{var_name}{subscript_py} = {expr_py}"

        # Function calls
        if typ == "call":
            _, name, args = ir
            if name == "loadMacros":
                return None
            py_args = [self._expr_to_py(a) for a in args]
            return f"{ind}{name}({', '.join(py_args)})"

        # Expressions
        if typ == "expr":
            _, expr = ir
            return f"{ind}{self._expr_to_py(expr)}"

        if typ == "bin":
            return f"{ind}{self._expr_to_py(ir)}"

        # Closure IR
        if typ == "closure":
            _, body_stmts = ir
            return self._emit_closure(body_stmts, indent)

        # Return statement
        if typ == "return":
            _, return_value = ir
            if return_value is None or (isinstance(return_value, str) and return_value == ""):
                return f"{ind}return"
            return_py = self._expr_to_py(return_value)
            return f"{ind}return {return_py}"

        # Parameter unpacking (nested within closure body)
        if typ == "param_unpack":
            _, var_list, deref_source = ir
            # Extract variable names
            var_names = []
            for var_tuple in var_list:
                var_name = self._desigil(var_tuple[1] if isinstance(
                    var_tuple, tuple) else var_tuple)
                var_names.append(var_name)

            # Emit as tuple unpacking assignment
            vars_str = ", ".join(var_names)
            deref_py = self._expr_to_py(deref_source)
            return f"{ind}{vars_str} = {deref_py}"

        # Unknown IR: produce raw comment
        return f"{ind}# {ir}"

    def _emit_block(self, block_ir: Any, indent: int) -> List[str]:
        """Emit a block of statements with proper indentation."""
        if block_ir[0] != "block":
            return []

        _, stmts = block_ir
        lines = []
        for stmt in stmts:
            emitted = self._emit_ir(stmt, indent)
            if emitted:
                lines.append(emitted)

        # Ensure block has at least pass if empty
        if not lines:
            lines.append("    " * indent + "pass")

        return lines

    def _emit_subscript(self, subscript_ir: Any) -> str:
        """Emit a subscript operation (array or hash)."""
        typ = subscript_ir[0]
        _, expr = subscript_ir
        expr_py = self._expr_to_py(expr)

        if typ == "array_subscript":
            return f"[{expr_py}]"
        elif typ == "hash_subscript":
            # Quote bare words if they're not already quoted
            if expr_py and expr_py[0] not in ('"', "'"):
                return f"['{expr_py}']"
            return f"[{expr_py}]"

        return f"[{expr_py}]"

    def _is_complex_closure(self, body_stmts: List[Any]) -> bool:
        """Check if closure requires extraction to def function.

        A closure is considered complex if it contains:
        - Control flow statements (if, unless, while, for, foreach, do_until)
        - Multiple statements (excluding parameter unpacking)

        Simple closures (single return statement) can be emitted as lambdas.
        """
        # Count non-parameter statements
        non_param_stmts = []
        has_control_flow = False

        for stmt in body_stmts:
            if isinstance(stmt, tuple):
                stmt_type = stmt[0]

                # Check for control flow
                if stmt_type in {"if", "unless", "while", "for", "foreach", "do_until"}:
                    has_control_flow = True

                # Track non-parameter statements
                if stmt_type != "param_unpack":
                    non_param_stmts.append(stmt)
            else:
                non_param_stmts.append(stmt)

        # Complex if has control flow or multiple non-param statements
        if has_control_flow:
            return True
        if len(non_param_stmts) > 1:
            return True

        return False

    def _generate_closure_name(self, context_name: str) -> str:
        """Generate unique function name for extracted closure.

        Uses a counter to ensure uniqueness across multiple closures.
        Example: _closure_checker_1, _closure_filter_2
        """
        self._closure_counter += 1
        # Use hex format for shorter names
        counter_hex = format(self._closure_counter, 'x')
        # Sanitize context name (keep only alphanumeric and underscore)
        safe_name = ''.join(c if c.isalnum() or c ==
                            '_' else '_' for c in context_name)
        safe_name = safe_name.strip('_') or 'closure'
        return f"_closure_{safe_name}_{counter_hex}"

    def _emit_closure(self, body_stmts: List[Any], indent: int, context_name: str = "func") -> Any:
        """Emit a Python function for a Perl closure (sub { ... }).

        Returns either a lambda string (for simple closures) or a tuple of
        (function_definition_lines, function_name) for complex closures.

        Simple closures (single return statement) are emitted as:
            lambda correct, student: (correct == student)

        Complex closures (with control flow or multiple statements) are
        extracted as separate def functions and return the function reference:
            Returns: (["def _closure_checker_1(...):", "    ..."], "_closure_checker_1")

        Args:
            body_stmts: List of IR tuples representing closure body
            indent: Current indentation level (for multi-line functions)
            context_name: Name of the variable being assigned (for function naming)

        Returns:
            str: Lambda expression for simple closures
            Tuple[List[str], str]: Function definition + reference for complex closures
        """
        # Extract parameters and body statements
        params = []
        body_for_return = []
        first_param_unpack = True

        for stmt in body_stmts:
            if isinstance(stmt, tuple) and stmt[0] == "param_unpack" and first_param_unpack:
                # Extract parameter names from FIRST param_unpack only
                # (subsequent ones are variable assignments in the body)
                _, var_list, deref_source = stmt
                # var_list is a list of ("var", "$name") tuples
                for var_tuple in var_list:
                    var_name = self._desigil(var_tuple[1] if isinstance(
                        var_tuple, tuple) else var_tuple)
                    params.append(var_name)
                first_param_unpack = False
            else:
                body_for_return.append(stmt)

        # Emit function body
        if not params:
            # No parameter unpacking found, use *args, **kwargs
            params_str = "*args, **kwargs"
        else:
            params_str = ", ".join(params)

        # Check if body is a single return statement
        if (len(body_for_return) == 1 and
            isinstance(body_for_return[0], tuple) and
                body_for_return[0][0] == "return"):
            # Single return - emit as lambda
            _, return_value = body_for_return[0]
            return_py = self._expr_to_py(return_value)
            return f"lambda {params_str}: {return_py}"

        # Complex closure - extract to separate def function
        if self._is_complex_closure(body_stmts):
            func_name = self._generate_closure_name(context_name)
            func_lines = []

            ind = "    " * indent

            # Function definition line
            func_lines.append(f"{ind}def {func_name}({params_str}):")

            # Emit body statements
            if body_for_return:
                python_keywords = {'return', 'if', 'else', 'elif',
                                   'for', 'while', 'def', 'class', 'import', 'from'}
                for stmt in body_for_return:
                    emitted = self._emit_ir(stmt, indent + 1)
                    if emitted:
                        if isinstance(emitted, list):
                            # Already a list of lines
                            func_lines.extend(emitted)
                        else:
                            # String - may contain embedded newlines from control flow statements
                            emitted_str = emitted
                            # Skip comments and malformed statements
                            emitted_stripped = emitted_str.strip()
                            if emitted_stripped.startswith("#"):
                                continue
                            # Skip assignments to Python keywords (e.g., "return = {}")
                            # Check first line only
                            if "=" in emitted_stripped.split("\n")[0]:
                                var_part = emitted_stripped.split("=")[
                                    0].strip()
                                if var_part in python_keywords:
                                    continue
                            # Split multi-line strings into individual lines
                            if "\n" in emitted_str:
                                func_lines.extend(emitted_str.split("\n"))
                            else:
                                func_lines.append(emitted_str)
                # If function has no body, add pass
                if len(func_lines) == 1:  # Just the def line
                    func_lines.append(f"{ind}    pass")
            else:
                # Empty body - add pass statement
                func_lines.append(f"{ind}    pass")

            # Filter out problematic lines (e.g., "return = {}")
            python_keywords = {'return', 'if', 'else', 'elif',
                               'for', 'while', 'def', 'class', 'import', 'from'}
            filtered_lines = []
            for line in func_lines:
                line_stripped = line.strip()
                # Skip lines that assign to Python keywords
                if "=" in line_stripped and not line_stripped.startswith("def ") and not line_stripped.startswith("if ") and not line_stripped.startswith("elif ") and not line_stripped.startswith("else"):
                    var_part = line_stripped.split("=")[0].strip()
                    if var_part in python_keywords:
                        continue
                filtered_lines.append(line)

            # Return tuple: (function_definition_lines, function_name)
            return (filtered_lines, func_name)

        # Fallback for closures that don't match patterns above
        return f"lambda {params_str}: None  # Complex Perl closure not fully translated"

    def _expr_to_py(self, expr: Any) -> str:
        """Lower an expression IR into a Python expression string."""
        if isinstance(expr, tuple):
            head = expr[0]

            # Variables
            if head == "var":
                return self._desigil(expr[1])

            # Binary operations
            if head == "bin":
                _, left, op, right = expr
                py_left = self._expr_to_py(left)
                py_right = self._expr_to_py(right)

                # Operator mapping
                op_map = {
                    "eq": "==", "ne": "!=", "lt": "<", "gt": ">",
                    "le": "<=", "ge": ">=",
                    ".": "+",  # String concatenation
                    "x": "*",  # String repetition
                    "||": " or ", "&&": " and ",
                    "or": " or ", "and": " and "
                }
                py_op = op_map.get(op, op)

                # Special handling for . and x operators
                # These could be either string operations or Vector dot/cross products
                # For now, use function calls that can handle both cases
                if op == ".":
                    # Could be string concatenation or dot product
                    # Use a helper function that tries dot first (for Vector)
                    # then falls back to string concatenation
                    return f"pg_concat({py_left}, {py_right})"
                elif op == "x":
                    # Could be string repetition or cross product
                    # Use a helper function that tries cross product first (for Vector)
                    # then falls back to string repetition
                    return f"pg_repeat({py_left}, {py_right})"

                return f"({py_left} {py_op} {py_right})"

            # Ternary operator
            if head == "ternary":
                _, cond, true_val, false_val = expr
                cond_py = self._expr_to_py(cond)
                true_py = self._expr_to_py(true_val)
                false_py = self._expr_to_py(false_val)
                return f"({true_py} if {cond_py} else {false_py})"

            # Range operator
            if head == "range":
                _, start, end = expr
                start_py = self._expr_to_py(start)
                end_py = self._expr_to_py(end)
                return f"range({start_py}, {end_py} + 1)"

            # Unary operations
            if head == "unary":
                _, op, operand = expr
                operand_py = self._expr_to_py(operand)
                if op == "!":
                    return f"(not {operand_py})"
                return f"({op}{operand_py})"

            # Postfix operations (method calls, subscripts)
            if head == "postfix":
                _, base, op = expr
                base_py = self._expr_to_py(base)

                # Handle Tree objects that weren't transformed yet
                if not isinstance(op, tuple):
                    try:
                        from lark import Tree
                        if isinstance(op, Tree):
                            # Transform the Tree to IR
                            op = self._transformer.transform(op)
                            # If it returned a list, take first item
                            if isinstance(op, list) and op:
                                op = op[0]
                    except:
                        pass

                # Now handle the postfix operation
                if isinstance(op, tuple):
                    if op[0] == "method_call":
                        _, method_name, args = op
                        # Special case: .reduce() is a property in Python MathObjects, not a method
                        # In Perl: ->reduce() and ->reduce are equivalent
                        # In Python: reduce is a @property, so .reduce() fails
                        if method_name == "reduce" and len(args) == 0:
                            return f"{base_py}.reduce"
                        # Special case: properties that shouldn't have parentheses
                        # These are Matrix/Vector properties that shouldn't be called as methods
                        property_names = {"transpose", "inverse", "norm",
                                          "dimensions", "trace", "det", "determinant", "value"}
                        if method_name in property_names and len(args) == 0:
                            return f"{base_py}.{method_name}"
                        # Special case: .with(...) needs to become .with_params(...) to avoid 'with' keyword
                        if method_name == "with":
                            method_name = "with_params"
                        arg_strs = []
                        for a in args:
                            if isinstance(a, tuple) and len(a) >= 2 and a[0] == "named_param":
                                # named_param: key => value
                                _, key_expr, val_expr = a
                                key_str = self._expr_to_py(key_expr)
                                val_str = self._expr_to_py(val_expr)
                                # Convert key to bareword if it's a variable
                                if isinstance(key_expr, tuple) and key_expr[0] == "var":
                                    # Just the variable name without sigil
                                    key_str = key_expr[1]
                                arg_strs.append(f"{key_str} = {val_str}")
                            else:
                                arg_strs.append(self._expr_to_py(a))

                        return f"{base_py}.{method_name}({', '.join(arg_strs)})"
                    elif op[0] in ("array_subscript", "hash_subscript"):
                        subscript_py = self._emit_subscript(op)
                        return f"{base_py}{subscript_py}"

                return base_py

            # Function calls
            if head == "call":
                _, name, args = expr
                # Convert Perl namespace operator :: to Python dot notation
                if isinstance(name, str):
                    name = name.replace("::", ".")

                # Special handling for ClassName.classMatch(obj, 'ClassName') -> isinstance(obj, ClassName)
                # This works for any class with a classMatch method (Value.classMatch, XYZ.classMatch, etc.)
                if name.endswith(".classMatch") and len(args) >= 2:
                    obj_expr = args[0]
                    class_name_expr = args[1]

                    # Extract the object to check
                    obj_py = self._expr_to_py(obj_expr)

                    # Extract the class name string (could be a string literal or variable)
                    class_name_py = self._expr_to_py(class_name_expr)

                    # Remove quotes if it's a string literal
                    # STRING tokens are returned as strings with quotes, e.g., '"Formula"' or "'Formula'"
                    if isinstance(class_name_py, str):
                        # Check if it's a quoted string (starts and ends with same quote)
                        if (class_name_py.startswith('"') and class_name_py.endswith('"')) or \
                           (class_name_py.startswith("'") and class_name_py.endswith("'")):
                            # Strip quotes to get the class name
                            class_name_str = class_name_py[1:-1]
                        else:
                            # Not a quoted string, use as-is (might be a variable)
                            class_name_str = class_name_py
                    else:
                        # Not a string, convert to string and try to extract
                        class_name_str = str(class_name_py).strip('"\'')

                    # Convert class name string to Python class name
                    # 'Formula' -> Formula, 'Real' -> Real, etc.
                    # These should be available from pg.mathobjects import *
                    return f"isinstance({obj_py}, {class_name_str})"

                arg_strings = []
                # Track if we have any string-key named params (only string literals)
                has_string_key_params = False
                string_key_params = []

                for a in args:
                    if isinstance(a, tuple) and len(a) >= 2 and a[0] == "named_param":
                        # named_param: key => value
                        _, key_expr, val_expr = a
                        # Check if key is SPECIFICALLY a string literal
                        # String literals are either ("string", ...) tuples or bare strings starting with quotes
                        is_string_literal = False
                        if isinstance(key_expr, tuple) and key_expr[0] == "string":
                            is_string_literal = True
                        elif isinstance(key_expr, str) and (key_expr.startswith('"') or key_expr.startswith("'")):
                            is_string_literal = True

                        if is_string_literal:
                            # String literal key like 'u(t)' - will create a dict
                            has_string_key_params = True
                            key_str = self._expr_to_py(
                                key_expr)  # Keep the quotes
                            val_str = self._expr_to_py(val_expr)
                            string_key_params.append((key_str, val_str))
                        else:
                            # Bareword or variable key - use as keyword argument
                            # For bareword: key_expr is just a token
                            # For variable: key_expr is ("var", name)
                            if isinstance(key_expr, tuple) and key_expr[0] == "var":
                                # Variable like $var
                                # Just the variable name without sigil
                                key_str = key_expr[1]
                            else:
                                # Bareword - convert to string
                                key_str = self._expr_to_py(key_expr)
                            val_str = self._expr_to_py(val_expr)
                            arg_strings.append(f"{key_str} = {val_str}")
                    else:
                        arg_strings.append(self._expr_to_py(a))

                # If we have string-key params, create a dict argument
                if has_string_key_params:
                    dict_items = [f"{k}: {v}" for k, v in string_key_params]
                    dict_str = "{" + ", ".join(dict_items) + "}"
                    # Insert dict as first positional argument
                    arg_strings.insert(0, dict_str)

                return f"{name}({', '.join(arg_strings)})"

            # Map and grep
            if head == "map":
                _, block_expr, list_expr = expr
                block_py = self._expr_to_py(block_expr)
                list_py = self._expr_to_py(list_expr)
                return f"[{block_py} for _ in {list_py}]"

            if head == "grep":
                _, block_expr, list_expr = expr
                block_py = self._expr_to_py(block_expr)
                list_py = self._expr_to_py(list_expr)
                return f"[_ for _ in {list_py} if {block_py}]"

            # Array dereferencing
            if head == "array_deref":
                _, var = expr
                var_py = self._expr_to_py(var)
                # In Python, arrays are already dereferenced, so just return the variable
                return var_py

            # Special variables like @_
            if head == "special_var":
                _, var = expr
                if var == "@_":
                    # @_ in Perl is the argument list - in Python closures this becomes the params
                    # We don't emit @_ directly; it's handled during parameter unpacking
                    return "args"  # Fallback - shouldn't reach here in normal cases
                return var

            # Array literals
            if head == "array":
                _, items = expr
                item_strs = [self._expr_to_py(item) for item in items]
                return f"[{', '.join(item_strs)}]"

            # Regex literal
            if head == "regex":
                _, pattern, flags = expr
                # Convert Perl regex flags to Python re flags
                flag_map = {
                    'i': 're.IGNORECASE',
                    'm': 're.MULTILINE',
                    's': 're.DOTALL',
                    'x': 're.VERBOSE',
                }
                py_flags = ' | '.join(flag_map.get(f, '')
                                      for f in str(flags) if f in flag_map)
                if py_flags:
                    return f're.compile(r"{pattern}", {py_flags})'
                else:
                    return f're.compile(r"{pattern}")'

            # Expression statements
            if head == "expr":
                return self._expr_to_py(expr[1])

            # Assignments (in expression context)
            if head == "assign":
                _, var, val = expr
                return f"{self._desigil(var[1])} = {self._expr_to_py(val)}"

        # Otherwise, treat as raw string and apply rewrite
        rewritten = self._rewrite_with_pygments(str(expr))
        return rewritten

    # ------------------------------------------------------------------
    # Pygments based rewriting
    # ------------------------------------------------------------------

    def _desigil(self, name: str) -> str:
        """Remove leading sigil characters from Perl variable names."""
        if name and name[0] in "$@%":
            return name[1:]
        return name

    def _convert_regexes(self, code: str) -> str:
        """Convert Perl qr/pattern/flags regexes to Python re.compile() calls."""
        import re as re_module

        # Pattern to match Perl regex literals: qr/pattern/flags
        # This handles patterns with escaped slashes inside
        pattern = r'qr/([^/]*(?:\\.[^/]*)*)/([imsxo]*)'

        def replace_regex(match):
            pattern_content = match.group(1)
            flags_str = match.group(2)
            # Convert Perl regex flags to Python re flags
            flag_map = {
                'i': 're.IGNORECASE',
                'm': 're.MULTILINE',
                's': 're.DOTALL',
                'x': 're.VERBOSE',
            }
            py_flags = ' | '.join(flag_map.get(f, '')
                                  for f in flags_str if f in flag_map)
            if py_flags:
                return f're.compile(r"{pattern_content}", {py_flags})'
            else:
                return f're.compile(r"{pattern_content}")'

        return re_module.sub(pattern, replace_regex, code)

    def _rewrite_with_pygments(self, code: str) -> str:
        """Fallback rewrite using Pygments for conservative token replacement."""
        # First, replace qr/pattern/flags with re.compile(pattern, flags)
        import re as re_module
        code = self._convert_regexes(code)

        tokens = list(self._perl_lexer.get_tokens_unprocessed(code))

        result: List[str] = []
        i = 0
        # Track brace context: True if in hash literal, False if in code block
        brace_context_stack: List[bool] = []
        # Track bracket context: nesting level of [...]
        bracket_depth = 0

        while i < len(tokens):
            _, ttype, text = tokens[i]
            # Preserve comments verbatim
            if ttype in Token.Comment:
                result.append(text)
                i += 1
                continue
            # Quote bare words before => in hash literals
            if ttype in Token.Name and brace_context_stack and brace_context_stack[-1]:
                # We're inside a hash literal, look ahead to see if next non-whitespace token is =>
                j = i + 1
                while j < len(tokens) and tokens[j][1] in Token.Text.Whitespace:
                    j += 1
                if j < len(tokens):
                    next_token = tokens[j]
                    # Check if it's => (either as two tokens = > or single =>)
                    is_fat_comma = False
                    if next_token[2] == '=>':
                        is_fat_comma = True
                    elif next_token[2] == '=' and j + 1 < len(tokens) and tokens[j + 1][2] == '>':
                        is_fat_comma = True

                    if is_fat_comma:
                        # Quote the bare word
                        result.append(f"'{text}'")
                        i += 1
                        continue
            # Handle false-positive regex tokens BEFORE string check (Regex is subclass of String)
            # Pygments misidentifies / division as regex
            if ttype in Token.Literal.String.Regex:
                # If it starts/ends with / and contains $variables, it might be division operators
                # Example: "/ 2, $b + $r * sqrt(2) /" is actually division, not regex
                if text.startswith('/') and text.endswith('/') and '$' in text:
                    # Strip the / delimiters and process as normal code
                    import re
                    content = text[1:-1] if len(text) > 2 else text
                    # Remove sigils from variables
                    converted = re.sub(
                        r'\$([a-zA-Z_][a-zA-Z0-9_]*)', r'\1', content)
                    # Re-add the / as division operators
                    result.append('/' + converted + '/')
                else:
                    result.append(text)
                i += 1
                continue
            # Handle qr/pattern/flags regex literals
            if text.startswith('qr/') and '/' in text[3:]:
                # Find the closing / and extract pattern and flags
                import re as re_module
                match = re_module.match(r'qr/([^/]*)/([imsxo]*)', text)
                if match:
                    pattern = match.group(1)
                    flags_str = match.group(2)
                    # Convert Perl regex flags to Python re flags
                    flag_map = {
                        'i': 're.IGNORECASE',
                        'm': 're.MULTILINE',
                        's': 're.DOTALL',
                        'x': 're.VERBOSE',
                    }
                    py_flags = ' | '.join(flag_map.get(f, '')
                                          for f in flags_str if f in flag_map)
                    if py_flags:
                        result.append(f're.compile(r"{pattern}", {py_flags})')
                    else:
                        result.append(f're.compile(r"{pattern}")')
                else:
                    result.append(text)
                i += 1
                continue
            # String interpolation: convert Perl "$var" to Python f"{var}"
            if ttype in Token.Literal.String:
                # Only process double-quoted strings (Perl interpolates these)
                if text.startswith('"') and '$' in text:
                    # Extract content without quotes
                    content = text[1:-1] if len(text) >= 2 else text
                    # Escape backslashes first (before processing braces)
                    # This handles LaTeX sequences like \( \) \Big etc.
                    escaped = content.replace('\\', '\\\\')
                    # Escape literal braces for f-strings
                    escaped = escaped.replace('{', '{{').replace('}', '}}')
                    # Convert $var to {var}
                    import re
                    converted = re.sub(
                        r'\$([a-zA-Z_][a-zA-Z0-9_]*)', r'{\1}', escaped)
                    result.append(f'f"{converted}"')
                else:
                    # For non-interpolated strings, escape backslashes
                    # This handles LaTeX in single-quoted strings
                    if '\\' in text and not text.startswith('r"') and not text.startswith("r'"):
                        # Extract quotes and content
                        if len(text) >= 2:
                            quote_char = text[0]
                            content = text[1:-1]
                            escaped_content = content.replace('\\', '\\\\')
                            result.append(
                                f'{quote_char}{escaped_content}{quote_char}')
                        else:
                            result.append(text)
                    else:
                        result.append(text)
                i += 1
                continue
            # Variables: handle $# array last index operator specially, then remove sigil
            if ttype in Token.Name.Variable:
                if text == '$#':
                    # $# followed by array name -> len(array_name) - 1
                    # Look ahead for the array name
                    j = i + 1
                    while j < len(tokens) and tokens[j][1] in Token.Text.Whitespace:
                        j += 1
                    if j < len(tokens) and tokens[j][1] in Token.Name.Variable:
                        array_token = tokens[j][2]
                        array_name = self._desigil(array_token)
                        result.append(f"len({array_name}) - 1")
                        i = j + 1  # Skip past the array name
                        continue
                    else:
                        # No array name following, just replace $# with an underscore to avoid syntax error
                        result.append("_")
                        i += 1
                        continue
                elif text.startswith('$#'):
                    # $#array_name (shouldn't happen with current Pygments, but handle it)
                    array_name = text[2:]
                    result.append(f"len({array_name}) - 1")
                    i += 1
                    continue
                else:
                    result.append(self._desigil(text))
                    i += 1
                    continue
            # Namespace tokens: convert :: to .
            if ttype in Token.Name.Namespace:
                # Pygments returns 'parser::Assignment' as a single token
                # Convert :: to .
                converted = text.replace('::', '.')
                result.append(converted)
                i += 1
                continue
            # Hash access: $h{key} -> h['key']
            # Also handle chained subscripts: Context()->{error}{msg} -> Context()['error']['msg']
            if text == '{' and i > 0:
                prev_token = tokens[i-1]
                # Handle if previous token is a variable, closing paren, closing brace, or >
                # (> indicates we just processed -> which was converted to .)
                if (prev_token[1] in Token.Name.Variable or
                    prev_token[2] == ')' or
                    prev_token[2] == '}' or
                        prev_token[2] == '>'):
                    # gather until closing brace
                    inner: List[str] = []
                    depth = 1
                    j = i + 1
                    while j < len(tokens) and depth > 0:
                        _, t2, s2 = tokens[j]
                        if s2 == '{':
                            depth += 1
                        elif s2 == '}':
                            depth -= 1
                            if depth == 0:
                                break
                        inner.append(s2)
                        j += 1
                    inner_text = ''.join(inner).strip()
                    # quote bare words if not already quoted
                    if inner_text and inner_text[0] not in "'\"":
                        inner_text = f"'{inner_text}'"
                    result.append('[' + inner_text + ']')
                    i = j + 1
                    continue
            # Combine operator pairs into single tokens for arrows, namespace
            # separators and fat comma.  Pygments splits '->' into two
            # separate Operator tokens '-' and '>' and likewise '=' and '>'
            # for '=>', and ':' and ':' for '::'.  Detect these pairs
            # here.
            if ttype in Token.Operator or ttype in Token.Punctuation:
                # Handle method arrow '->'
                if text == '-' and i + 1 < len(tokens) and tokens[i+1][2] == '>':
                    i += 2
                    # Skip any empty tokens after ->
                    while i < len(tokens) and tokens[i][2] == '':
                        i += 1
                    # Check what comes after ->
                    if i < len(tokens):
                        next_idx, next_ttype, next_text = tokens[i]
                        # If next token is {, it's a hash subscript - don't append . (let [ handle it)
                        if next_text == '{':
                            # Don't append anything, let the hash subscript code handle {
                            continue
                        # Otherwise append . for method/property access
                        result.append('.')
                        # If it's a method/property name
                        if next_ttype in Token.Name or next_ttype == Token.Operator.Word:
                            # Look ahead to see what follows (skip empty tokens)
                            j = i + 1
                            while j < len(tokens) and tokens[j][2] == '':
                                j += 1
                            has_parens = False
                            has_arrow = False
                            has_brace = False
                            if j < len(tokens):
                                lookahead_idx, lookahead_ttype, lookahead_text = tokens[j]
                                if lookahead_text == '(':
                                    has_parens = True
                                elif lookahead_text == '-' and j + 1 < len(tokens) and tokens[j + 1][2] == '>':
                                    has_arrow = True  # Another -> follows, so this is property access
                                elif lookahead_text == '{':
                                    has_brace = True  # Hash subscript follows
                            # Always append the method/property name
                            result.append(next_text)
                            # Known properties that shouldn't have parentheses (from MathObject/Matrix/Vector)
                            property_names = {
                                "transpose", "inverse", "norm", "dimensions", "trace", "det", "determinant", "reduce", "value"}
                            # Only add () if no parens AND no arrow AND no brace (i.e., final method in chain)
                            # AND it's not a known property
                            if not has_parens and not has_arrow and not has_brace and next_text not in property_names:
                                result.append('()')
                            i += 1
                            continue
                    else:
                        # Nothing after ->, just append .
                        result.append('.')
                    continue
                # Handle namespace separator '::'
                if text == ':' and i + 1 < len(tokens) and tokens[i+1][2] == ':':
                    result.append('.')
                    i += 2
                    continue
                # Handle fat comma '=>'
                if text == '=' and i + 1 < len(tokens) and tokens[i+1][2] == '>':
                    # Inside hash literal, treat as dict colon
                    if brace_context_stack and brace_context_stack[-1]:
                        result.append(': ')
                    # Inside bracket (list) context, quote the key and use comma
                    elif bracket_depth > 0:
                        # Need to quote the previous bareword if it exists
                        # Look back to find it
                        j = len(result) - 1
                        while j >= 0 and result[j].strip() == '':
                            j -= 1
                        if j >= 0:
                            # Check if previous token looks like a bareword
                            prev = result[j].strip()
                            if prev and prev.isidentifier() and prev not in ('True', 'False', 'None'):
                                # Replace it with quoted version
                                result[j] = f'"{prev}"'
                        result.append(', ')
                    else:
                        # Check if previous token is ] or ) - then it's a pair separator
                        j = len(result) - 1
                        while j >= 0 and result[j].strip() == '':
                            j -= 1
                        if j >= 0 and result[j].strip() in (']', ')'):
                            # It's separating elements, use comma
                            result.append(', ')
                        else:
                            # Otherwise it's an assignment
                            result.append(' = ')
                    i += 2
                    continue
                # Handle Perl string concatenation operator '.'
                # In Perl: "str" . "ing" concatenates strings
                # In Python: "str" + "ing"
                # We need to distinguish from method access: obj.method()
                if text == '.':
                    # Check if this is string concatenation (binary operator) or method access
                    # Look at previous and next tokens to determine context
                    is_concat = False
                    if i > 0 and i + 1 < len(tokens):
                        # Skip whitespace and empty tokens before
                        j = i - 1
                        while j >= 0 and (tokens[j][1] in Token.Text.Whitespace or tokens[j][2] == ''):
                            j -= 1
                        # Skip whitespace and empty tokens after
                        k = i + 1
                        while k < len(tokens) and (tokens[k][1] in Token.Text.Whitespace or tokens[k][2] == ''):
                            k += 1

                        if j >= 0 and k < len(tokens):
                            prev_text = tokens[j][2]
                            prev_ttype = tokens[j][1]
                            next_text = tokens[k][2]
                            next_ttype = tokens[k][1]

                            # Special case: .with_params( is a method call, never concatenation
                            if next_text == 'with_params':
                                is_concat = False
                            else:
                                # It's concatenation if there's whitespace around the dot
                                # In Perl: "str" . "ing" has spaces
                                # In Perl: obj.method() has no spaces
                                # Check if preceded by string, closing paren, or ends with expression
                                is_expr_before = (
                                    prev_ttype in Token.Literal.String or
                                    prev_text == ')' or
                                    prev_text == ']' or
                                    prev_ttype in Token.Literal.Number
                                )
                                # Check if followed by string, function call, or expression
                                is_expr_after = (
                                    next_ttype in Token.Literal.String or
                                    next_ttype in Token.Literal.Number or
                                    next_ttype in Token.Name  # Function call like ans_rule()
                                )
                                # It's concatenation if both sides look like expressions
                                # and we have whitespace (i != j+1 or k != i+1)
                                if is_expr_before and is_expr_after:
                                    # Check for whitespace
                                    has_space_before = (j < i - 1)
                                    has_space_after = (k > i + 1)
                                    if has_space_before or has_space_after:
                                        is_concat = True

                    if is_concat:
                        result.append(' + ')
                        i += 1
                        continue
            # Fat comma '=>' collapsed as a single text (rare)
            if text == '=>':
                # Inside hash literal, treat as dict colon
                if brace_context_stack and brace_context_stack[-1]:
                    result.append(': ')
                # Inside bracket (list) context, quote the key and use comma
                elif bracket_depth > 0:
                    # Need to quote the previous bareword if it exists
                    j = len(result) - 1
                    while j >= 0 and result[j].strip() == '':
                        j -= 1
                    if j >= 0:
                        prev = result[j].strip()
                        if prev and prev.isidentifier() and prev not in ('True', 'False', 'None'):
                            result[j] = f'"{prev}"'
                    result.append(', ')
                else:
                    # Check if previous token is ] or ) - then it's a pair separator
                    j = len(result) - 1
                    while j >= 0 and result[j].strip() == '':
                        j -= 1
                    if j >= 0 and result[j].strip() in (']', ')'):
                        result.append(', ')
                    else:
                        result.append(' = ')
                i += 1
                continue
            # Track bracket context for list literals
            if text == '[':
                bracket_depth += 1
            elif text == ']':
                bracket_depth = max(0, bracket_depth - 1)

            # Track brace context for hash literal detection
            if text == '{':
                # Determine if this is a hash literal or code block
                # Hash literals typically follow: =>, (, [, ,, or start of line
                is_hash_literal = False
                if i > 0:
                    # Look back for previous non-whitespace token
                    j = i - 1
                    while j >= 0 and tokens[j][1] in Token.Text.Whitespace:
                        j -= 1
                    if j >= 0:
                        prev_token = tokens[j][2]
                        # Hash literal indicators
                        if prev_token in ('=>', '>', '(', '[', ',', '='):
                            is_hash_literal = True
                else:
                    # At start, assume hash literal
                    is_hash_literal = True
                brace_context_stack.append(is_hash_literal)
            elif text == '}':
                if brace_context_stack:
                    brace_context_stack.pop()
            # Drop trailing semicolon at end
            if text == ';' and i == len(tokens) - 1:
                i += 1
                continue
            result.append(text)
            i += 1

        rewritten = ''.join(result).rstrip()

        # Post-processing: Apply additional transformations
        import re

        # Convert .with( to .with_params( to avoid Python 'with' keyword
        # At this point, -> has already been converted to . by IR emission
        # This must happen AFTER binary expr emission because . is treated as binary operator
        rewritten = re.sub(r'\.with\(', '.with_params(', rewritten)

        # Convert .reduce() to .reduce (property, not method in Python MathObjects)
        # In Perl: ->reduce() and ->reduce are equivalent
        # In Python: reduce is a @property, so calling it with () fails
        rewritten = re.sub(r'\.reduce\(\)', '.reduce', rewritten)

        # Convert Perl range operator .. to Python range()
        # This is tricky because .. can appear in various contexts: (a..b), [ a..b ], a..b
        # The Lark parser handles most cases, but fallback is needed for complex expressions
        # Strategy: Convert all .. operators to range, then simplify any redundant arithmetic

        # First: Convert all range operators. These patterns handle the most common cases.
        # Pattern: token .. token where tokens are: numbers, variables, len() expressions

        # Handle: 0 .. len(array) - 1
        rewritten = re.sub(
            r'(\d+)\s*\.\.\s*(len\([^)]*\)\s*-\s*1)',
            r'range(\1, \2 + 1)',
            rewritten
        )

        # Handle: len(array) - 1 .. number
        rewritten = re.sub(
            r'(len\([^)]*\)\s*-\s*1)\s*\.\.\s*(\d+)',
            r'range(\1, \2 + 1)',
            rewritten
        )

        # Handle: number .. number
        rewritten = re.sub(
            r'(\d+)\s*\.\.\s*(\d+)',
            r'range(\1, \2 + 1)',
            rewritten
        )

        # Handle: $#array .. number or number .. $#array
        rewritten = re.sub(
            r'(\$?#[a-zA-Z_]\w*)\s*\.\.\s*(\d+)',
            r'range(\1, \2 + 1)',
            rewritten
        )
        rewritten = re.sub(
            r'(\d+)\s*\.\.\s*(\$?#[a-zA-Z_]\w*)',
            r'range(\1, \2 + 1)',
            rewritten
        )

        # Handle: $#array .. $#array2 or similar
        rewritten = re.sub(
            r'(\$?#[a-zA-Z_]\w*)\s*\.\.\s*(\$?#[a-zA-Z_]\w*)',
            r'range(\1, \2 + 1)',
            rewritten
        )

        # Second: Simplify redundant arithmetic in range() calls
        # When we have range(X, Y - 1 + 1), simplify to range(X, Y)
        rewritten = re.sub(
            r'range\(([^,]+),\s*(len\([^)]*\))\s*-\s*1\s*\+\s*1\)',
            r'range(\1, \2)',
            rewritten
        )

        # Handle: range(X, expr - N + N) -> range(X, expr) for any constant N
        # This catches cases like: range(0, len(array) - 1 + 1)
        def simplify_range_arithmetic(match):
            """Simplify range() calls with canceling arithmetic."""
            prefix = match.group(1)
            end_expr = match.group(2)

            # Check if end_expr is like "... - N + N" where N is the same number
            # Extract the subtracted and added numbers
            subtract_match = re.search(r'-\s*(\d+)\s*\+\s*(\d+)$', end_expr)
            if subtract_match:
                sub_num = subtract_match.group(1)
                add_num = subtract_match.group(2)
                if sub_num == add_num:
                    # Remove the " - N + N" part
                    simplified = re.sub(r'\s*-\s*' + re.escape(sub_num) +
                                        r'\s*\+\s*' + re.escape(add_num) + r'$', '', end_expr)
                    return f"range({prefix}, {simplified})"

            return match.group(0)

        rewritten = re.sub(
            r'range\(([^,]+),\s*([^)]*-\s*\d+\s*\+\s*\d+)\)',
            simplify_range_arithmetic,
            rewritten
        )

        # Convert Perl string repetition operator 'x' to Python '*'
        # Pattern: (string/variable) x (number) or (variable) x (variable)
        # Be careful not to convert 'x' when it's a variable name or part of identifiers
        rewritten = re.sub(r'(["\'\)])\s+x\s+(\d+)', r'\1 * \2', rewritten)
        rewritten = re.sub(
            r'(\b[a-zA-Z_]\w*)\s+x\s+(\d+)', r'\1 * \2', rewritten)
        # Also handle variable x variable (e.g., v3 x v4 for cross product)
        rewritten = re.sub(
            r'(\b[a-zA-Z_]\w*)\s+x\s+([a-zA-Z_]\w*\b)', r'\1 * \2', rewritten)

        # Condense spaces around equals from fat comma conversion
        rewritten = re.sub(r'\s+=\s+', ' = ', rewritten)

        # Convert string comparison operators
        # Be careful: these are only operators when surrounded by expressions/values,
        # not variable names. Look for operators preceded/followed by closing parens,
        # numbers, closing brackets, or the words 'not', 'and', 'or'
        # Pattern: (expression) OPERATOR (expression)
        rewritten = re.sub(r'([)\]\w])\s+eq\s+', r'\1 == ', rewritten)
        rewritten = re.sub(r'([)\]\w])\s+ne\s+', r'\1 != ', rewritten)
        rewritten = re.sub(r'([)\]\w])\s+lt\s+', r'\1 < ', rewritten)
        rewritten = re.sub(r'([)\]\w])\s+gt\s+', r'\1 > ', rewritten)
        rewritten = re.sub(r'([)\]\w])\s+le\s+', r'\1 <= ', rewritten)
        rewritten = re.sub(r'([)\]\w])\s+ge\s+', r'\1 >= ', rewritten)

        # Convert ternary operator: cond ? true : false  -->  true if cond else false
        # This is complex because ternaries can appear in various contexts.
        # The key insight: We need to find ternaries at the RIGHT nesting level.
        # For example: `foo(bar, x > 0 ? 1 : 2)` should convert the ternary INSIDE the call.
        def convert_ternaries(text: str) -> str:
            """Recursively convert ternary operators, handling nesting correctly."""
            if '?' not in text or ':' not in text:
                return text

            # Find all ? positions and their matching : at the same paren/bracket depth
            depth = 0
            ternary_positions = []  # List of (question_pos, colon_pos) tuples

            i = 0
            while i < len(text):
                ch = text[i]
                if ch in '([{':
                    depth += 1
                elif ch in ')]}':
                    depth -= 1
                elif ch == '?' and depth >= 0:
                    # Found a ?, now find its matching :
                    question_depth = depth
                    j = i + 1
                    local_depth = depth
                    while j < len(text):
                        if text[j] in '([{':
                            local_depth += 1
                        elif text[j] in ')]}':
                            local_depth -= 1
                        elif text[j] == ':' and local_depth == question_depth:
                            ternary_positions.append((i, j))
                            break
                        j += 1
                i += 1

            # Process ternaries from innermost (rightmost) to outermost
            # This handles nested ternaries correctly
            for question_pos, colon_pos in reversed(ternary_positions):
                # Extract the parts
                # Need to find where this ternary starts (the condition)
                # Work backwards from ? to find the start of the condition
                # The condition starts after the previous operator or delimiter

                # Find the start of the condition by working backwards
                cond_start = 0
                depth = 0
                for k in range(question_pos - 1, -1, -1):
                    if text[k] in ')]}':
                        depth += 1
                    elif text[k] in '([{':
                        depth -= 1
                        if depth < 0:
                            # Hit an unmatched opening bracket
                            cond_start = k + 1
                            break
                    elif depth == 0 and text[k] in ',;=(':
                        # Hit a delimiter at depth 0
                        cond_start = k + 1
                        break

                # Find the end of the false value
                # Work forward from : to find where it ends
                false_end = len(text)
                depth = 0
                for k in range(colon_pos + 1, len(text)):
                    if text[k] in '([{':
                        depth += 1
                    elif text[k] in ')]}':
                        depth -= 1
                        if depth < 0:
                            # Hit an unmatched closing bracket
                            false_end = k
                            break
                    elif depth == 0 and text[k] in ',;)':
                        # Hit a delimiter at depth 0
                        false_end = k
                        break

                cond = text[cond_start:question_pos].strip()
                true_val = text[question_pos + 1:colon_pos].strip()
                false_val = text[colon_pos + 1:false_end].strip()

                # Build the replacement
                replacement = f"{true_val} if ({cond}) else {false_val}"

                # Replace in the text
                text = text[:cond_start] + replacement + text[false_end:]

                # Only process one ternary at a time, then restart
                # (because positions change after replacement)
                if len(ternary_positions) > 1:
                    return convert_ternaries(text)

            return text

        if '?' in rewritten and ':' in rewritten:
            rewritten = convert_ternaries(rewritten)

        # Convert logical operators
        rewritten = rewritten.replace('||', ' or ')
        rewritten = rewritten.replace('&&', ' and ')

        # Convert Perl $#array (last index) to len(array)-1
        rewritten = re.sub(r'\$\#\$([a-zA-Z_]\w*)', r'len(\1)-1', rewritten)
        rewritten = re.sub(r'\$\#([a-zA-Z_]\w*)', r'len(\1)-1', rewritten)

        # Convert Perl reference operator ~~& to just the function name
        rewritten = re.sub(r'~~&([a-zA-Z_]\w*)', r'\1', rewritten)
        rewritten = re.sub(r'~~([a-zA-Z_]\w*)', r'\1', rewritten)

        # Special case: Wrap CapitalizedWord(...) = "string" patterns in parens for tuple pairs
        # This happens with AnswerHints( Formula(...) => "msg", ... )
        rewritten = re.sub(
            r'(?<![a-z])([A-Z][a-zA-Z0-9_]*\([^)]*\))\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')',
            r'(\1, \2)',
            rewritten,
        )

        # string literal = value (legacy keyword style) → positional args
        rewritten = re.sub(
            r'"([^"]*?)"\s*=\s*(["\'][^"\']*["\'])',
            r'"\1", \2',
            rewritten,
        )
        rewritten = re.sub(
            r"'([^']*?)'\s*=\s*([\"'][^\"']*[\"'])",
            r"'\1', \2",
            rewritten,
        )

        # Context().functions.add(name => { ... }) expects positional string key
        rewritten = re.sub(
            r'(\.add\(\s*)([a-z_][a-zA-Z0-9_]*)\s*=\s*\{',
            r"\1'\2', {",
            rewritten,
        )

        return rewritten

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

    def _transform_text_block(self, content: str) -> str:
        r"""
        Transform TEXT block content to Python expression(s).

        Handles:
        - Variable interpolation: $a → ", a, "
        - Function calls: \{ ans_rule(20) \} → ", ans_rule(20), "
        - LaTeX math: \( ... \) → keep as-is
        - Special vars: $PAR → ", PAR(), "

        Returns:
            Python expression string suitable for TEXT() call
        """
        import re
        segments: List[str] = []
        pos = 0
        while pos < len(content):
            var_match = re.search(r'\$([a-zA-Z_][a-zA-Z0-9_]*)', content[pos:])
            func_match = re.search(r'\\{([^}]+)\\}', content[pos:])
            next_var_pos = pos + var_match.start() if var_match else len(content)
            next_func_pos = pos + func_match.start() if func_match else len(content)
            if next_var_pos < next_func_pos:
                if next_var_pos > pos:
                    text_segment = content[pos:next_var_pos]
                    segments.append(repr(text_segment))
                var_name = var_match.group(1)
                if var_name in ('PAR', 'BR', 'BBOLD', 'EBOLD', 'BITALIC', 'EITALIC', 'BCENTER', 'ECENTER', 'BUL', 'EUL'):
                    segments.append(f"{var_name}()")
                else:
                    segments.append(f"str({var_name})")
                pos = next_var_pos + len(var_match.group(0))
            elif next_func_pos < len(content):
                if next_func_pos > pos:
                    text_segment = content[pos:next_func_pos]
                    segments.append(repr(text_segment))
                func_code = func_match.group(1).strip()
                transformed_code = self._compile_expr(func_code)
                segments.append(transformed_code)
                pos = next_func_pos + len(func_match.group(0))
            else:
                if pos < len(content):
                    text_segment = content[pos:]
                    segments.append(repr(text_segment))
                break
        if not segments:
            return '""'
        return ', '.join(segments)

    def _escape_triple_quotes(self, text: str) -> str:
        """Escape special characters in text for Python triple quoted string literals."""
        text = text.replace("\\", "\\\\")
        text = text.replace("'''", "\'\'\'")
        text = text.replace('"""', '\"\"\"')
        return text

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

    def _transform_pgml_evaluators(self, pgml_content: str) -> str:
        """Transform Perl syntax to Python in PGML evaluator expressions."""
        import re
        result: List[str] = []
        i = 0
        while i < len(pgml_content):
            if pgml_content[i] == '{':
                brace_depth = 1
                j = i + 1
                while j < len(pgml_content) and brace_depth > 0:
                    if pgml_content[j] == '{':
                        brace_depth += 1
                    elif pgml_content[j] == '}':
                        brace_depth -= 1
                        if brace_depth == 0:
                            break
                    j += 1
                if brace_depth == 0:
                    code_block = pgml_content[i+1:j]
                    # Convert Perl syntax to Python:
                    # - => to = (Perl fat comma to Python assignment)
                    # - -> to . (Perl method call to Python attribute access)
                    # - :: to . (Perl package separator to Python module separator)
                    # - Remove $ from variable names
                    transformed = code_block.replace(
                        '=>', '=').replace('->', '.').replace('::', '.')
                    transformed = re.sub(
                        r'\$([a-zA-Z_]\w*)', r'\1', transformed)
                    result.append('{')
                    result.append(transformed)
                    result.append('}')
                    i = j + 1
                    continue
            result.append(pgml_content[i])
            i += 1
        return ''.join(result)

    def _fix_reference_dereferences(self, pg_source: str) -> str:
        """Fix Perl reference dereference arrows for parsing.

        Converts:
            $x->[$i] to $x[$i]
            $x->{key} to $x{key}
            $#$x to len(x) - 1

        This must happen before parsing because the grammar doesn't have rules for
        ->[ or ->{ patterns, and it doesn't understand $# for array length.
        """
        # Fix $x->[$i] pattern (dereference to array subscript)
        # Also handle $x->[expr] patterns
        pg_source = re.sub(
            r'\$(\w+(?:\[[^\]]*\])*)\s*->\s*\[',
            r'$\1[',
            pg_source
        )

        # Fix $x->{key} pattern (dereference to hash subscript)
        pg_source = re.sub(
            r'\$(\w+(?:\[[^\]]*\])*)\s*->\s*\{',
            r'$\1{',
            pg_source
        )

        # Fix $#$x pattern (array length - returns last index)
        # $#$arr becomes len(arr) - 1
        pg_source = re.sub(
            r'\$#\$(\w+)',
            r'(len(\1) - 1)',
            pg_source
        )

        # Fix @$x pattern (array dereference)
        # @$arr becomes list($arr)
        pg_source = re.sub(
            r'@\$(\w+)',
            r'list($\1)',
            pg_source
        )

        return pg_source

    def _convert_heredocs_global(self, pg_source: str) -> str:
        """Convert Perl heredocs (<<END_MARKER) and qq/.../  to Python triple-quoted strings at the source level.

        Processes the entire source before line splitting to handle heredocs properly.

        Converts:
            HEADER_TEXT(MODES(TeX => '', HTML => <<END_STYLE));
            <style>...</style>
            END_STYLE

        To a form like:
            HEADER_TEXT(MODES(TeX => '', HTML => '''<style>

            </style>'''))

        Also converts:
            $var = qq/content here/;

        To:
            $var = '''content here''';

        Note: The result will be re-split into lines by the caller, so embedded newlines
        in the triple-quoted strings are preserved.
        """
        import re as re_module

        lines = pg_source.split('\n')
        result_lines: List[str] = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check if this line contains qq/.../ or qq{...} or similar (Perl quoted strings)
            # qq/ = qq with / delimiters
            # qq{ = qq with { } delimiters
            # etc.
            qq_match = re_module.search(r'qq([/\{\[\(\|])', line)
            if qq_match:
                # Found a qq construct
                delimiter = qq_match.group(1)
                # Determine the closing delimiter
                closing_delim = {
                    '[': ']', '{': '}', '(': ')', '/': '/', '|': '|'}.get(delimiter, delimiter)

                # Split the line at the qq start
                before = line[:qq_match.start()]
                # Skip the 'qq' and opening delimiter
                after_qq_start = line[qq_match.end():]

                # Collect content until we find the closing delimiter
                content_lines: List[str] = []
                remaining = after_qq_start

                # Check if closing delimiter is on the same line
                close_idx = remaining.find(closing_delim)
                if close_idx != -1:
                    # Found closing delimiter on same line
                    content = remaining[:close_idx]
                    after_content = remaining[close_idx + 1:]

                    # Escape content for Python triple-quoted string
                    content = content.replace('\\', '\\\\')
                    content = content.replace("'''", "\\'''")

                    new_line = f"{before}'''{content}'''{after_content}"
                    result_lines.append(new_line)
                else:
                    # Closing delimiter is on a later line
                    content_lines.append(remaining)
                    i += 1
                    while i < len(lines):
                        current = lines[i]
                        close_idx = current.find(closing_delim)
                        if close_idx != -1:
                            # Found the closing delimiter
                            content_lines.append(current[:close_idx])
                            after_content = current[close_idx + 1:]
                            break
                        else:
                            content_lines.append(current)
                        i += 1

                    # Build the replacement: join content with actual newlines
                    content = '\n'.join(content_lines)
                    # Escape content for Python triple-quoted string
                    content = content.replace('\\', '\\\\')
                    content = content.replace("'''", "\\'''")

                    new_line = f"{before}'''{content}'''{after_content}"
                    result_lines.append(new_line)
            # Check for traditional heredoc (<<MARKER)
            elif re_module.search(r'<<([A-Z_][A-Z0-9_]*)', line):
                heredoc_match = re_module.search(r'<<([A-Z_][A-Z0-9_]*)', line)
                marker = heredoc_match.group(1)

                # Split the line at the heredoc marker
                before = line[:heredoc_match.start()]
                # Everything after <<MARKER on same line
                after_marker = line[heredoc_match.end():]

                # Collect the content until we find the marker on its own line
                content_lines: List[str] = []
                i += 1
                while i < len(lines):
                    current = lines[i]
                    # Check if this line is just the marker (with optional whitespace)
                    if re_module.match(rf'^\s*{re_module.escape(marker)}\s*$', current):
                        break
                    content_lines.append(current)
                    i += 1

                # Build the replacement: join content with actual newlines
                content = '\n'.join(content_lines)
                # Escape backslashes in the content (for regex characters, etc.)
                content = content.replace('\\', '\\\\')
                # Escape triple quotes
                content = content.replace("'''", "\\'''")

                # Build the new line with triple-quoted string
                # Include the content with embedded newlines
                new_line = f"{before}'''{content}'''{after_marker}"
                result_lines.append(new_line)
            else:
                result_lines.append(line)

            i += 1

        return '\n'.join(result_lines)

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
