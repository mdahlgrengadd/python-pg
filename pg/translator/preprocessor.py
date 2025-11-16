"""
⚠️  DEPRECATED - DO NOT USE THIS FILE ⚠️

THIS PREPROCESSOR HAS BEEN REPLACED BY pg_preprocessor_pygment.py

The regex-based preprocessor in this file is DEPRECATED and should NOT be used.
It has been replaced by the Lark grammar-based preprocessor in pg_preprocessor_pygment.py
which provides proper parsing and transformation of PG/Perl syntax to Python.

ALWAYS use PygmentPreprocessor from pg.preprocessor_pygment.py instead.

DO NOT fallback to this preprocessor under any circumstances.

This file is kept only for historical reference and will be removed in a future version.

---

OLD DESCRIPTION (for reference only):

PG Preprocessor - Transform PG syntactic sugar.

Handles:
- BEGIN_TEXT...END_TEXT → text accumulation
- BEGIN_PGML...END_PGML → PGML rendering
- BEGIN_SOLUTION...END_SOLUTION → solution text
- BEGIN_HINT...END_HINT → hint text
- Comment removal
- Backslash handling

Reference: Translator.pm::default_preprocess_code() (lines 1348-1378)

⚠️  DEPRECATED - USE pg_preprocessor_pygment.py INSTEAD ⚠️
"""

import re
from pathlib import Path
from dataclasses import dataclass

from .pgml_parser import PGMLParser, PGMLRenderer


@dataclass
class PreprocessResult:
    """Result of preprocessing a PG file.

    ⚠️ DEPRECATED - Use pg_preprocessor_pygment.PreprocessResult instead
    """

    code: str
    """Preprocessed Python code"""

    text_blocks: list[tuple[str, str]]
    """List of (block_type, content) for TEXT, PGML, SOLUTION, HINT blocks"""

    line_map: dict[int, int]
    """Map from preprocessed line number to original line number"""


class PGPreprocessor:
    """
    ⚠️  DEPRECATED - DO NOT USE ⚠️

    This class has been replaced by PygmentPreprocessor in pg_preprocessor_pygment.py

    Use PygmentPreprocessor instead - it provides proper Lark grammar-based parsing
    of PG/Perl syntax and correct transformation to Python.

    This regex-based preprocessor has known bugs and should not be used.

    ---

    OLD DESCRIPTION (for reference):

    Preprocess PG files to transform syntactic sugar into executable Python.

    PG files use Perl-like syntax with special blocks:
    - BEGIN_TEXT...END_TEXT: Problem statement
    - BEGIN_PGML...END_PGML: PGML markup
    - BEGIN_SOLUTION...END_SOLUTION: Solution text
    - BEGIN_HINT...END_HINT: Hint text

    This preprocessor transforms these into Python function calls that
    accumulate text in the execution environment.

    ⚠️  DEPRECATED - USE PygmentPreprocessor INSTEAD ⚠️
    """

    # Block markers
    BLOCK_PATTERNS = {
        "TEXT": (r"BEGIN_TEXT\s*$", r"^END_TEXT"),
        "PGML": (r"BEGIN_PGML\s*$", r"^END_PGML"),
        "SOLUTION": (r"BEGIN_SOLUTION\s*$", r"^END_SOLUTION"),
        "HINT": (r"BEGIN_HINT\s*$", r"^END_HINT"),
        "PGML_SOLUTION": (r"BEGIN_PGML_SOLUTION\s*$", r"^END_PGML_SOLUTION"),
        "PGML_HINT": (r"BEGIN_PGML_HINT\s*$", r"^END_PGML_HINT"),
        "TIKZ": (r"BEGIN_TIKZ\s*$", r"^END_TIKZ"),
    }

    def preprocess(self, pg_source: str, use_sandbox_macros: bool = True) -> PreprocessResult:
        """
        ⚠️ DEPRECATED - Use PygmentPreprocessor.preprocess() instead ⚠️

        Preprocess PG source code.

        Args:
            pg_source: Raw PG file content
            use_sandbox_macros: If True, skip generating imports (sandbox provides them)

        Returns:
            PreprocessResult with transformed code and metadata

        WARNING: This method uses regex-based transformation and has known bugs.
        Use PygmentPreprocessor from pg.preprocessor_pygment.py instead.
        """
        import warnings
        warnings.warn(
            "PGPreprocessor is deprecated. Use PygmentPreprocessor from "
            "pg_preprocessor_pygment.py instead. This regex-based preprocessor "
            "has known bugs and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )

        lines = pg_source.split("\n")
        output_lines: list[str] = []
        text_blocks: list[tuple[str, str]] = []
        line_map: dict[int, int] = {}

        # First pass: collect all loadMacros() calls to generate imports
        import_lines: list[str] = []
        loaded_macros_comment = None

        if not use_sandbox_macros:
            # Generate imports only if not using sandbox macros
            for line in lines:
                if "loadMacros" in line:
                    # Extract macro file names
                    match = re.search(r'loadMacros\((.*?)\)', line, re.DOTALL)
                    if match:
                        imports, comment = self._transform_load_macros(
                            match.group(1))
                        import_lines.extend(imports)
                        loaded_macros_comment = comment

        # Track if we've inserted imports yet
        imports_inserted = False

        # Track if we're inside a multiline loadMacros() call
        in_load_macros = False
        paren_depth = 0

        i = 0
        while i < len(lines):
            original_line = lines[i]

            # Join multi-line continuations (Perl allows implicit continuations)
            # If line ends with = or , and next line is indented, join them
            # Skip this for comment lines
            is_comment = original_line.lstrip(' \t').startswith('#')
            if not is_comment:
                # Don't join if this is a } else { line - it should stay on its own line
                if not re.match(r'^\s*\}\s*else\s*\{\s*$', original_line):
                    while i + 1 < len(lines):
                        stripped = original_line.rstrip()
                        next_line = lines[i + 1] if i + 1 < len(lines) else ""

                        # Don't join if next line is a comment
                        if next_line.lstrip(' \t').startswith('#'):
                            break

                        # Don't join TO a } else { line either!
                        if re.match(r'^\s*\}\s*else\s*\{\s*$', next_line):
                            break

                        # Check if this looks like a continuation
                        should_join = False

                        # Case 1: Line ends with = or , or ( or [
                        # Strip inline comments first to check actual ending
                        stripped_no_comment = self._strip_inline_comment(
                            stripped)
                        if stripped_no_comment and stripped_no_comment[-1] in '=,([':
                            if next_line and next_line[0] in ' \t':
                                should_join = True

                        # Case 2: Next line starts with binary operator (., +, -, etc.) for continuation
                        # This handles Perl string concatenation: "str" \n . "more"
                        if not should_join:
                            next_stripped = next_line.lstrip(' \t')
                            if next_stripped and next_stripped[0] in '.+-':
                                # Make sure it's not a unary minus or method call
                                if next_stripped[0] == '.' or (next_stripped[0] in '+-' and len(next_stripped) > 1 and next_stripped[1] in ' \t"\''):
                                    should_join = True

                        # Case 2b: Next line starts with -> (Perl method chaining)
                        # This handles: $obj = Func(...) \n ->method(...)
                        if not should_join:
                            next_stripped = next_line.lstrip(' \t')
                            if next_stripped and next_stripped.startswith('->'):
                                should_join = True

                        # Case 3: Unmatched parentheses/brackets (check without comments)
                        if not should_join:
                            check_line = self._strip_inline_comment(stripped)
                            open_count = check_line.count(
                                '(') + check_line.count('[') + check_line.count('{')
                            close_count = check_line.count(
                                ')') + check_line.count(']') + check_line.count('}')
                            if open_count > close_count:
                                should_join = True

                        if should_join and next_line.strip():
                            # Strip inline comment from current line before joining
                            # to prevent comment from eating subsequent joined content
                            line_without_comment = self._strip_inline_comment(
                                original_line.rstrip())
                            original_line = line_without_comment + \
                                ' ' + next_line.lstrip(' \t')
                            i += 1
                        else:
                            break

            output_line_num = len(output_lines) + 1

            # Track line mapping
            line_map[output_line_num] = i + 1

            # Handle compound statements (e.g., DOCUMENT(); loadMacros(...); TEXT(...))
            # Split by semicolon but preserve the parts that aren't loadMacros
            if ';' in original_line and 'loadMacros' in original_line:
                parts = original_line.split(';')
                non_loadmacros_parts = []
                skip_rest_of_line = False

                for part in parts:
                    part = part.strip()
                    if 'loadMacros' in part:
                        # Check if this is a complete loadMacros call or start of multi-line
                        if '(' in part and part.count('(') == part.count(')'):
                            # Complete on this part, skip it
                            continue
                        else:
                            # Multi-line loadMacros starts here
                            in_load_macros = True
                            paren_depth = part.count('(') - part.count(')')
                            skip_rest_of_line = True
                            break
                    else:
                        # Keep non-loadMacros parts
                        if part:
                            non_loadmacros_parts.append(part)

                # If we have non-loadMacros parts, output them
                if non_loadmacros_parts:
                    combined = '; '.join(non_loadmacros_parts)
                    if combined:
                        # Process the combined line through normal transformation
                        transformed = self._transform_line(combined)
                        output_lines.append(transformed)

                i += 1
                continue

            # Check if we're entering a standalone loadMacros() call
            if 'loadMacros' in original_line and '(' in original_line:
                in_load_macros = True
                # Count opening and closing parens on this line
                paren_depth = original_line.count(
                    '(') - original_line.count(')')

                # If balanced on same line, skip it and move on
                if paren_depth == 0:
                    in_load_macros = False
                    i += 1
                    continue
                else:
                    # Multi-line loadMacros - skip this line and continue tracking
                    i += 1
                    continue

            # If we're inside a loadMacros() call, track parentheses
            if in_load_macros:
                paren_depth += original_line.count('(') - \
                    original_line.count(')')
                if paren_depth <= 0:
                    # End of loadMacros() call
                    in_load_macros = False
                i += 1
                continue

            # Check if this is DOCUMENT() - insert imports right after it (if needed)
            if not imports_inserted and re.match(r'^\s*DOCUMENT\(\s*\)', original_line):
                # Handle compound statements like: DOCUMENT(); loadMacros(...); TEXT(...)
                # Split by semicolon and process each part
                if ';' in original_line:
                    parts = original_line.split(';')
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        # Skip loadMacros parts (macros pre-loaded in sandbox)
                        if 'loadMacros' in part:
                            continue
                        # Transform and add other parts
                        if part:
                            transformed = self._transform_line(part)
                            if transformed:
                                output_lines.append(transformed)
                else:
                    output_lines.append(original_line)

                # Insert imports after DOCUMENT() only if not using sandbox
                if import_lines:
                    output_lines.append("")  # Blank line
                    output_lines.extend(import_lines)
                    if loaded_macros_comment:
                        output_lines.append(loaded_macros_comment)
                    output_lines.append("")  # Blank line
                imports_inserted = True
                i += 1
                continue

            # Check for Perl closures: sub { ... } - stub them out
            # These are typically used for custom answer checkers
            # Example: checker => sub { ... }
            sub_match = re.search(r'(=>|=)\s*sub\s*\{', original_line)
            if sub_match:
                # Found start of a sub {} closure
                # Track brace depth to find the end
                closure_start_idx = i
                closure_lines = [original_line]
                brace_depth = original_line.count(
                    '{') - original_line.count('}')

                # Collect all lines of the closure
                i += 1
                while i < len(lines) and brace_depth > 0:
                    current_line = lines[i]
                    closure_lines.append(current_line)
                    brace_depth += current_line.count(
                        '{') - current_line.count('}')
                    i += 1

                # Now replace the entire sub { ... } with lambda: None
                first_line = closure_lines[0]

                # Check if this is part of a function parameter (key => sub { ... })
                param_match = re.search(r'(\w+)\s*=>\s*sub\s*\{', first_line)
                if param_match:
                    param_name = param_match.group(1)
                    # Find where the sub starts
                    sub_start = first_line.find('sub')
                    # Keep everything before 'sub'
                    prefix = first_line[:sub_start]
                    # Replace sub { ... } with lambda: None
                    # Check if there's more content after the closure on the last line
                    last_line = closure_lines[-1] if closure_lines else ""

                    # Find the closing } and any suffix
                    suffix = ""
                    close_brace_match = re.search(r'\}(.*)$', last_line)
                    if close_brace_match:
                        suffix = close_brace_match.group(1)

                    # Create the stubbed line and transform it
                    stubbed_line = f"{prefix}lambda *args, **kwargs: None{suffix}"
                    transformed = self._transform_line(stubbed_line)
                    output_lines.append(
                        f"{transformed}  # Stubbed Perl closure")
                else:
                    # Assignment form: $var = sub { ... }
                    assign_match = re.search(
                        r'(\w+)\s*=\s*sub\s*\{', first_line)
                    if assign_match:
                        var_name = assign_match.group(1)
                        indent = re.match(r'^(\s*)', first_line).group(1)
                        stubbed_line = f"{indent}{var_name} = lambda *args, **kwargs: None"
                        transformed = self._transform_line(stubbed_line)
                        output_lines.append(
                            f"{transformed}  # Stubbed Perl closure")
                    else:
                        # Unknown form, comment it out
                        output_lines.append(
                            f"# {first_line}  # Skipped Perl closure")

                continue

            # Check for do { ... } until (condition) loops
            do_until_match = re.match(r'^\s*do\s*\{', original_line)
            if do_until_match:
                # Check if this is a single-line do-until
                # Match with or without parentheses around condition
                single_line_until = re.search(
                    r'\}\s*until\s*(?:\(([^)]+)\)|(.+))$', original_line)

                if single_line_until:
                    # Single-line do-until: do { body } until (condition) or do { body } until condition
                    condition = single_line_until.group(
                        1) or single_line_until.group(2)
                    condition = condition.strip()

                    # Transform condition
                    condition = self._transform_line(condition)

                    # Extract body between { and }
                    body_match = re.search(r'do\s*\{([^}]+)\}', original_line)
                    if body_match:
                        body = body_match.group(1).strip()
                        transformed_body = self._transform_line(body)

                        # Generate Python while loop with post-test
                        # Perl: do {...} until (condition) means repeat until condition is TRUE
                        output_lines.append(f'while True:')
                        output_lines.append(f'    {transformed_body}')
                        output_lines.append(f'    if ({condition}):')
                        output_lines.append(f'        break')

                        i += 1
                        continue

                # Multi-line do-until: collect the block
                block_lines = [original_line]
                brace_depth = original_line.count(
                    '{') - original_line.count('}')
                i += 1

                # Collect lines until we find the matching }
                while i < len(lines) and brace_depth > 0:
                    line = lines[i]
                    block_lines.append(line)
                    brace_depth += line.count('{') - line.count('}')
                    i += 1

                # Now check if the last line has "until (condition)" or "until condition"
                last_line = block_lines[-1] if block_lines else ""
                # Match with or without parentheses around condition
                until_match = re.search(
                    r'\}\s*until\s*(?:\(([^)]+)\)|(.+))$', last_line)

                # If not found and we have a next line, check if "until" is on the next line
                if not until_match and i < len(lines):
                    next_line = lines[i]
                    # Check if next line starts with "until"
                    until_match = re.match(
                        r'^\s*until\s*(?:\(([^)]+)\)|([^;]+))', next_line)
                    if until_match:
                        # Add the next line to block_lines
                        block_lines.append(next_line)
                        i += 1

                if until_match:
                    # Get condition from either group 1 (with parens) or group 2 (without)
                    condition = until_match.group(1) or until_match.group(2)
                    condition = condition.strip()

                    # Transform condition (convert Perl operators)
                    condition = self._transform_line(condition)

                    # Extract the body (everything between do { and } until)
                    body_lines = []
                    # First line: remove "do {" at start (use regex to avoid removing 'do' from 'random')
                    first = re.sub(r'^\s*do\s*\{', '', block_lines[0]).strip()
                    if first:
                        body_lines.append(first)

                    # Check if last line is just "until" (on separate line from })
                    last_line_is_until = re.match(
                        r'^\s*until\s+', block_lines[-1])

                    if last_line_is_until:
                        # Middle lines: all lines except first and last (skip the "until" line)
                        for line in block_lines[1:-1]:
                            # Remove trailing } if present
                            cleaned = re.sub(r'\s*}\s*$', '', line).strip()
                            if cleaned:
                                body_lines.append(cleaned)
                    else:
                        # Middle lines: add as-is
                        for line in block_lines[1:-1]:
                            body_lines.append(line)

                        # Last line: remove "} until (...)" or "} until condition"
                        last = re.sub(
                            r'\}\s*until\s*.*$', '', block_lines[-1]).strip()
                        if last:
                            body_lines.append(last)

                    # Transform body lines
                    transformed_body = []
                    for line in body_lines:
                        # Strip existing indentation and transform
                        stripped_line = line.lstrip()
                        transformed = self._transform_line(
                            stripped_line.rstrip())
                        if transformed:
                            # Add consistent 4-space indentation
                            transformed_body.append('    ' + transformed)

                    # Generate Python while loop with post-test (like do-until)
                    # Perl: do {...} until (condition) means repeat until condition is TRUE
                    # Python: while True: ... if (condition): break
                    output_lines.append('while True:')
                    output_lines.extend(transformed_body)
                    output_lines.append(f'    if ({condition}):')
                    output_lines.append(f'        break')

                    continue
                else:
                    # Not a proper do-until, fall through to normal processing
                    i = i - len(block_lines) + 1

            # Check for multi-line Perl for-loops: for my? $VAR (EXPR) { ... }
            # Must handle BEFORE line-by-line conversion to preserve block structure
            # Variable can have $ prefix (Perl) or not (already transformed)
            for_loop_match = re.match(
                r'^\s*for\s+(?:my\s+)?(?:\$)?([a-zA-Z_]\w*)\s*\(([^)]+)\)\s*\{', original_line)
            if for_loop_match:
                var = for_loop_match.group(1)
                expr = for_loop_match.group(2)

                # Check if single-line: for my $k (0..$n) { stmt; }
                single_line_match = re.search(r'\{([^}]+)\}', original_line)
                if single_line_match and original_line.count('{') == 1 and original_line.count('}') == 1:
                    # Single-line for-loop
                    body = single_line_match.group(1).strip()
                    # Transform expression (handle Perl range: START .. END)
                    # Variables can have $ prefix (e.g., $n) or not
                    expr = re.sub(r'(\d+|(?:\$)?[a-zA-Z_]\w*)\s*\.\.\s*(\d+|(?:\$)?[a-zA-Z_]\w*)',
                                  lambda m: f'range({m.group(1).strip()}, {m.group(2).strip()}+1)',
                                  expr)
                    # Transform the expression (strips $ prefixes, converts operators)
                    expr = self._transform_line(expr)
                    transformed_body = self._transform_line(body)
                    output_lines.append(f'for {var} in {expr}:')
                    output_lines.append(f'    {transformed_body}')
                    i += 1
                    continue

                # Multi-line for-loop: collect the block
                block_lines = [original_line]
                brace_depth = original_line.count(
                    '{') - original_line.count('}')
                i += 1

                # Collect lines until we find the matching }
                while i < len(lines) and brace_depth > 0:
                    line = lines[i]
                    block_lines.append(line)
                    brace_depth += line.count('{') - line.count('}')
                    i += 1

                # Transform the expression (handle Perl range: START .. END)
                # Variables can have $ prefix (e.g., $n) or not
                expr = re.sub(r'(\d+|(?:\$)?[a-zA-Z_]\w*)\s*\.\.\s*(\d+|(?:\$)?[a-zA-Z_]\w*)',
                              lambda m: f'range({m.group(1).strip()}, {m.group(2).strip()}+1)',
                              expr)
                # Transform the expression (strips $ prefixes, converts operators)
                expr = self._transform_line(expr)

                # Extract body lines
                body_lines = []
                # First line: remove "for ... {" part
                first = re.sub(
                    r'^\s*for\s+(?:my\s+)?\w+\s*\([^)]+\)\s*\{', '', block_lines[0]).strip()
                if first:
                    body_lines.append(first)

                # Middle lines: all lines except first and last
                for line in block_lines[1:-1]:
                    body_lines.append(line)

                # Last line: remove closing }
                if len(block_lines) > 1:
                    last = re.sub(r'\s*}\s*$', '', block_lines[-1]).strip()
                    if last:
                        body_lines.append(last)

                # Transform body lines
                transformed_body = []
                for line in body_lines:
                    stripped_line = line.lstrip()
                    if stripped_line:
                        # Split by semicolon to handle multiple statements per line
                        statements = [s.strip()
                                      for s in stripped_line.split(';') if s.strip()]
                        for stmt in statements:
                            transformed = self._transform_line(stmt)
                            if transformed:
                                # Add consistent 4-space indentation
                                transformed_body.append('    ' + transformed)

                # Generate Python for-loop
                output_lines.append(f'for {var} in {expr}:')
                output_lines.extend(transformed_body)

                continue

            # Check for inline if-else statements: if (cond) { stmt; } else { stmt; }
            # Python requires these on separate lines with indentation
            # Pattern: if (condition) { statements } else { statements }
            # DISABLED: This pattern causes issues with complex multi-line if-else blocks
            # that get joined. Multi-line blocks should be handled line-by-line.
            inline_if_match = None  # Temporarily disable
            # inline_if_match = re.match(r'^(\s*)if\s*\(([^)]+)\)\s*\{(.+)\}\s*else\s*\{(.+)\}\s*$', original_line)
            if False and inline_if_match:
                import sys
                print(f"DEBUG: inline if-else matched!", file=sys.stderr)
                print(f"  Line length: {len(original_line)}", file=sys.stderr)
                print(
                    f"  else_body length: {len(inline_if_match.group(4))}", file=sys.stderr)
                print(
                    f"  else_body: {inline_if_match.group(4)!r}", file=sys.stderr)

                indent = inline_if_match.group(1)
                condition = inline_if_match.group(2).strip()
                if_body = inline_if_match.group(3).strip()
                else_body = inline_if_match.group(4).strip()

                # Transform condition and bodies
                transformed_condition = self._transform_line(condition)

                # Split bodies by semicolon
                if_statements = [s.strip()
                                 for s in if_body.split(';') if s.strip()]
                else_statements = [s.strip()
                                   for s in else_body.split(';') if s.strip()]

                # Generate Python if-else block
                output_lines.append(f'{indent}if ({transformed_condition}):')
                for stmt in if_statements:
                    transformed_stmt = self._transform_line(stmt)
                    output_lines.append(f'{indent}    {transformed_stmt}')
                output_lines.append(f'{indent}else:')
                for stmt in else_statements:
                    transformed_stmt = self._transform_line(stmt)
                    output_lines.append(f'{indent}    {transformed_stmt}')

                i += 1
                continue

            # Check for inline if statements (no else): if (cond) { stmt; }
            # Note: This regex must handle braces inside strings, so we use a more robust pattern
            # that matches to the last } on the line, not the first one
            inline_if_only_match = re.match(
                r'^(\s*)if\s*\(([^)]+)\)\s*\{(.+)\}\s*$', original_line)
            if inline_if_only_match:
                indent = inline_if_only_match.group(1)
                condition = inline_if_only_match.group(2).strip()
                if_body = inline_if_only_match.group(3).strip()

                # Transform condition and body
                transformed_condition = self._transform_line(condition)

                # Split body by semicolon
                if_statements = [s.strip()
                                 for s in if_body.split(';') if s.strip()]

                # Generate Python if block
                output_lines.append(f'{indent}if ({transformed_condition}):')
                for stmt in if_statements:
                    transformed_stmt = self._transform_line(stmt)
                    output_lines.append(f'{indent}    {transformed_stmt}')

                i += 1
                continue

            # Check for inline for-loop statements: for VAR in EXPR: stmt1; stmt2; ...
            # Python requires these on separate lines with indentation
            inline_for_match = re.match(
                r'^(\s*)(for\s+\w+\s+in\s+[^:]+):\s*(.+)', original_line)
            if inline_for_match:
                indent = inline_for_match.group(1)
                for_header = inline_for_match.group(2)
                inline_body = inline_for_match.group(3)

                # Split inline body by semicolon (respecting strings and parentheses)
                def split_statements(text):
                    """Split by semicolon but respect strings and parentheses."""
                    statements = []
                    current = []
                    in_string = False
                    string_char = None
                    paren_depth = 0
                    bracket_depth = 0

                    j = 0
                    while j < len(text):
                        char = text[j]

                        # Handle string boundaries
                        if char in ('"', "'") and (j == 0 or text[j-1] != '\\'):
                            if not in_string:
                                in_string = True
                                string_char = char
                            elif char == string_char:
                                in_string = False
                                string_char = None

                        # Track depths (not in strings)
                        if not in_string:
                            if char == '(':
                                paren_depth += 1
                            elif char == ')':
                                paren_depth -= 1
                            elif char == '[':
                                bracket_depth += 1
                            elif char == ']':
                                bracket_depth -= 1
                            elif char == ';' and paren_depth == 0 and bracket_depth == 0:
                                # Found a statement boundary
                                stmt = ''.join(current).strip()
                                if stmt:
                                    statements.append(stmt)
                                current = []
                                j += 1
                                continue

                        current.append(char)
                        j += 1

                    # Last statement
                    stmt = ''.join(current).strip()
                    if stmt:
                        statements.append(stmt)

                    return statements

                statements = split_statements(inline_body)

                # Only convert if there are multiple statements or semicolons
                # (Single statement can stay on one line in Python)
                if len(statements) > 1 or ';' in inline_body:
                    # Transform the for header
                    transformed_header = self._transform_line(for_header)
                    output_lines.append(f'{indent}{transformed_header}:')

                    # Transform each statement and add with indentation
                    for stmt in statements:
                        transformed_stmt = self._transform_line(stmt)
                        if transformed_stmt:
                            output_lines.append(
                                f'{indent}    {transformed_stmt}')

                    i += 1
                    continue

            # Check for block markers
            block_found = False
            for block_type, (begin_pattern, end_pattern) in self.BLOCK_PATTERNS.items():
                if re.search(begin_pattern, original_line):
                    # Found block start
                    block_content_lines: list[str] = []
                    i += 1  # Move past BEGIN line

                    # Collect block content until END marker
                    while i < len(lines):
                        if re.match(end_pattern, lines[i]):
                            break
                        block_content_lines.append(lines[i])
                        i += 1

                    # Join content
                    block_content = "\n".join(block_content_lines)

                    # Store block
                    text_blocks.append((block_type, block_content))

                    # Generate Python code for this block
                    if "PGML" in block_type:
                        # PGML blocks - render at runtime with context
                        # Store the PGML content and render it during execution
                        # BUT FIRST: Transform Perl syntax to Python in evaluator expressions
                        block_content = self._transform_pgml_evaluators(
                            block_content)

                        block_var = f"PGML_BLOCK_{len(text_blocks) - 1}"
                        escaped_content = self._escape_triple_quotes(
                            block_content)
                        output_lines.append(
                            f"{block_var} = '''\\n{escaped_content}\\n'''"
                        )
                        # Call PGML renderer (will be available in sandbox)
                        if "SOLUTION" in block_type:
                            output_lines.append(f"SOLUTION(PGML({block_var}))")
                        elif "HINT" in block_type:
                            output_lines.append(f"HINT(PGML({block_var}))")
                        else:
                            output_lines.append(f"TEXT(PGML({block_var}))")
                    elif block_type == "TIKZ":
                        # TIKZ blocks - raw strings with backslashes preserved
                        # TikZ contains TeX/TikZ code that should NOT be transformed
                        # Store as raw string to preserve backslashes
                        block_var = f"TIKZ_BLOCK_{len(text_blocks) - 1}"
                        # Use raw string (r'''...''') to preserve backslashes
                        escaped_content = block_content.replace(
                            "'''", r"\'\'\'")
                        output_lines.append(
                            f"{block_var} = r'''\\n{escaped_content}\\n'''"
                        )
                        # TikZ blocks are typically assigned to a variable or method
                        # The previous line should have the assignment target
                        # For now, just skip output - the raw string is stored
                        # output_lines.append(f"# TikZ block stored in {block_var}")
                    else:
                        # Plain TEXT blocks - convert to TEXT() calls
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

            if not block_found:
                # Check if line contains do { } until/while mid-line (after semicolon)
                if re.search(r'do\s*\{[^}]*\}\s*(until|while)\s+', original_line):
                    # Split by semicolon, process do-until separately
                    parts = original_line.split(';')
                    for part_idx, part in enumerate(parts):
                        part = part.strip()
                        if not part:
                            continue

                        # Check if this part has do-until/do-while
                        do_match = re.match(
                            r'do\s*\{([^}]+)\}\s*(until|while)\s+(.+)', part)
                        if do_match:
                            body = do_match.group(1).strip()
                            loop_type = do_match.group(2)
                            condition = do_match.group(3).strip()

                            # Transform body and condition
                            transformed_body = self._transform_line(body)
                            transformed_condition = self._transform_line(
                                condition)

                            # Generate while True loop with break
                            output_lines.append('while True:')
                            output_lines.append(f'    {transformed_body}')
                            if loop_type == 'until':
                                # until COND means: break if COND is true
                                output_lines.append(
                                    f'    if ({transformed_condition}):')
                            else:  # while
                                # while COND means: break if COND is false
                                output_lines.append(
                                    f'    if not ({transformed_condition}):')
                            output_lines.append(f'        break')
                        else:
                            # Regular part - transform normally
                            transformed = self._transform_line(part)
                            if transformed:
                                output_lines.append(transformed)
                else:
                    # Regular line - pass through with transformations
                    transformed = self._transform_line(original_line)
                    output_lines.append(transformed)

            i += 1

        code = "\n".join(output_lines)
        return PreprocessResult(code=code, text_blocks=text_blocks, line_map=line_map)

    def _transform_line(self, line: str) -> str:
        """
        Transform a single line of PG code.

        Handles:
        - loadMacros() → skip (handled in first pass)
        - Perl variable syntax: $var → var
        - Array syntax: @array → array
        - Hash access: $hash{key} → hash['key']
        - Comment removal (# comments)
        - Semicolon removal (optional in Python)
        """
        import re

        # Handle } else { followed by content → split into else: and content on next line
        # This happens when line joining joins } else { with the next line
        else_match = re.match(r'^(\s*)\}\s*else\s*\{\s*(.+)$', line)
        if else_match:
            # Return just "else:" and let the content be handled separately
            # But we can't split it here since we return a single line
            # Instead, we'll need to NOT join these lines in the first place
            # For now, just handle the transformation
            indent = else_match.group(1)
            rest = else_match.group(2)
            # Transform the rest and return as "else: <content>" but that's not valid Python
            # We need to return multiple lines, but _transform_line returns a single string
            # WORKAROUND: Return with embedded newline
            transformed_rest = self._transform_line(rest)
            return f'{indent}else:\n{indent}    {transformed_rest}'

        # Handle } else { → else:
        else_bracket_match = re.match(r'^\s*\}\s*else\s*\{\s*$', line)
        if else_bracket_match:
            # Don't preserve indentation - else should be at same level as if
            return 'else:'

        # Handle multi-line if statements: if (...) { → if (...)  :
        # This handles if blocks that span multiple lines (body on next line)
        if_block_match = re.match(
            r'^(\s*)(if|elsif|while|for|foreach|until|unless)\s*\(([^)]+)\)\s*\{\s*(.*)$', line, re.IGNORECASE)
        if if_block_match:
            indent = if_block_match.group(1)
            keyword = if_block_match.group(2).lower()
            condition = if_block_match.group(3)
            rest = if_block_match.group(4).strip()

            # Convert elsif → elif
            if keyword == 'elsif':
                keyword = 'elif'
            # Convert unless → if not
            if keyword == 'unless':
                keyword = 'if not'

            # If there's content after the {, we need to handle it differently
            # For now, just convert the opening line
            if rest:
                # Single-line block with content: if (...) { stmt
                # This will be handled by inline-if pattern or needs special handling
                # For now, convert to: if (...): stmt
                transformed_condition = self._transform_line(condition)
                transformed_rest = self._transform_line(rest)
                return f'{indent}{keyword} ({transformed_condition}):\n{indent}    {transformed_rest}'
            else:
                # Multi-line block: if (...) {
                # Convert to: if (...):
                transformed_condition = self._transform_line(condition)
                return f'{indent}{keyword} ({transformed_condition}):'

        # Handle closing braces (convert to pass or remove)
        if re.match(r'^\s*\}\s*$', line):
            return ''  # Remove standalone closing braces

        # Skip loadMacros() - already handled in first pass
        if 'loadMacros' in line:
            # If it's on the same line as other code, remove just the loadMacros call
            if ';' in line:
                # Split by semicolon, remove loadMacros parts
                parts = line.split(';')
                cleaned_parts = [p for p in parts if 'loadMacros' not in p]
                if cleaned_parts:
                    line = ';'.join(cleaned_parts).strip()
                    if not line:
                        return ""
                else:
                    return ""
            else:
                # Entire line is loadMacros
                return ""

        # Handle DOCUMENT() and ENDDOCUMENT() - keep as-is
        if re.match(r'^\s*(DOCUMENT|ENDDOCUMENT)\(\s*\)', line):
            return line

        # Transform hash access: $hash{key} → hash['key']
        # Match $var{...} and convert to var['...']
        line = re.sub(
            r'\$([a-zA-Z_][a-zA-Z0-9_]*)\{([^}]+)\}', r"\1['\2']", line)

        # Transform Perl array variables: @array → array
        line = re.sub(r'@([a-zA-Z_][a-zA-Z0-9_]*)', r'\1', line)

        # Transform Perl string interpolation: "$var text" → f"{var} text"
        # Find all double-quoted strings and convert those with $var to f-strings
        def convert_string_interpolation(match):
            quote_char = match.group(1)  # " or '
            content = match.group(2)

            # Only convert double-quoted strings (Perl interpolates these)
            if quote_char == '"':
                # Check if contains $var
                if '$' in content:
                    # First, escape literal braces that should remain as-is
                    # In f-strings, { and } need to be {{ and }} if they're literal
                    # We need to escape braces BEFORE converting $var to {var}
                    escaped_content = content.replace(
                        '{', '{{').replace('}', '}}')

                    # Now convert $var to {var} - these will be unescaped single braces
                    new_content = re.sub(
                        r'\$([a-zA-Z_][a-zA-Z0-9_]*)', r'{\1}', escaped_content)
                    return f'f"{new_content}"'

            # Return as-is for single quotes or strings without variables
            return match.group(0)

        # Match strings carefully (handle escaped quotes)
        line = re.sub(r'(["\'])([^\1]*?)\1',
                      convert_string_interpolation, line)

        # Transform Perl $#array (last index): $#arr → len(arr)-1
        # Must do BEFORE stripping $ sigils
        # Example: random(0, $#functions) → random(0, len(functions)-1)
        line = re.sub(r'\$\#([a-zA-Z_][a-zA-Z0-9_]*)', r'len(\1)-1', line)

        # Transform Perl reference operator: ~~&func → func
        # Example: install_problem_grader(~~&custom_grader) → install_problem_grader(custom_grader)
        # In Perl, ~~& creates a reference to a subroutine; in Python, just use the function name
        line = re.sub(r'~~&([a-zA-Z_][a-zA-Z0-9_]*)', r'\1', line)

        # Transform Perl scalar variables: $var → var (outside of strings now)
        # Use negative lookbehind to avoid matching in strings
        line = re.sub(r'\$([a-zA-Z_][a-zA-Z0-9_]*)', r'\1', line)

        # Transform Perl logical operators: || → or, && → and
        # These are used in conditionals: if (x != 0 || y != 0)
        # Be careful not to match inside strings (already handled by string protection)
        line = line.replace('||', ' or ')
        line = line.replace('&&', ' and ')

        # Transform Perl method call operator: -> → .
        # Special case: ->with( becomes .with_params( to avoid Python keyword
        # But preserve method names like ->withPostFilter(, ->withUnitsFor(, etc.
        line = line.replace('->with(', '.with_params(')
        # Don't split method names starting with 'with' - they're valid Python identifiers
        line = line.replace('->', '.')

        # Add parentheses to Perl method calls that don't have them
        # In Perl: $obj->method is equivalent to $obj->method()
        # In Python: obj.method() is a call, obj.method is a property
        # Pattern: .method_name followed by whitespace, semicolon, closing paren/bracket, or end of line
        # Match: .cmp; or .cmp) or .cmp at end of line
        # Don't match: .reduce() (already has parens) or .key (hash access)
        # Common PG methods without parens: cmp, eval, TeX, string, value, etc.
        line = re.sub(
            r'\.([a-zA-Z_][a-zA-Z0-9_]*)(?=\s*[;,)\]\}]|\s*$)', r'.\1()', line)

        # Remove empty parentheses after methods that should be properties
        # In Perl, ->reduce() and ->reduce are equivalent
        # In Python, we made these properties, so remove the ()
        line = re.sub(r'\.reduce\(\)', '.reduce', line)

        # Transform chained hash access: .{key} or ){key} → ['key']
        # After -> to . conversion, Context()->{error}{msg} becomes Context().{error}.{msg}
        # We need to convert .{key} and ){key} patterns to ['key']
        # This handles: Context().{error} → Context()['error']
        #               obj.{key1}.{key2} → obj['key1']['key2']
        # Handle quoted keys: {'key'} → ['key'] (don't double-quote)
        # Need to capture the dot to remove it: .{key} → ['key'] not .['key']
        def transform_hash_access(match):
            prefix = match.group(1)
            key = match.group(2)
            # Check if key is already quoted
            if (key.startswith("'") and key.endswith("'")) or (key.startswith('"') and key.endswith('"')):
                bracket = f"[{key}]"
            else:
                bracket = f"['{key}']"
            # If prefix is '.', replace it; otherwise keep it
            if prefix == '.':
                return bracket
            else:
                return prefix + bracket

        while True:
            new_line = re.sub(r"([.\)\]'])\{([^}]+)\}",
                              transform_hash_access, line)
            if new_line == line:
                break
            line = new_line

        # Transform Perl hash/dict operator: => → = (for kwargs) or : (for dict literals)
        # Context-dependent transformation:
        # - Inside {} braces: => becomes : for dict literals
        # - In function calls: name => value becomes name=value for keyword arguments
        # - As array separator: => becomes ,
        def replace_hash_arrow(text: str) -> str:
            """Replace => appropriately based on context."""
            result = []
            i = 0
            in_string = False
            string_char = None
            escaped = False
            brace_depth = 0  # Track {} braces for dict literals

            while i < len(text):
                char = text[i]

                if escaped:
                    result.append(char)
                    escaped = False
                    i += 1
                    continue

                if char == '\\':
                    result.append(char)
                    escaped = True
                    i += 1
                    continue

                if char in ('"', "'"):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None
                    result.append(char)
                    i += 1
                    continue

                # Track brace depth outside strings
                if not in_string:
                    if char == '{':
                        brace_depth += 1
                    elif char == '}':
                        brace_depth -= 1

                # Check for => outside of strings
                if not in_string and i + 1 < len(text) and text[i:i+2] == '=>':
                    # Inside {} braces: dict literal, use :
                    # Outside braces: keyword argument, use =
                    if brace_depth > 0:
                        result.append(':')
                    else:
                        result.append('=')
                    i += 2
                    continue

                result.append(char)
                i += 1

            return ''.join(result)

        line = replace_hash_arrow(line)

        # Special case: ] => [ pattern (array fat comma in argument lists)
        # This is Perl's way of creating pairs: [ arr1 ] => [ arr2 ]
        # Transform to tuple syntax: ], [ which becomes ([ arr1 ], [ arr2 ]) in function args
        # Must do this AFTER general => replacement to override it
        line = re.sub(r'\]\s*=\s*\[', '], [', line)

        # Special case: name = value inside array literals (from => in arrays)
        # Perl: [ 'text', key => value ] becomes [ 'text', key = value ] (invalid Python)
        # Fix: Transform to [ 'text', {'key': value} ]
        # Pattern: , identifier = value inside [ ... ]
        # First try: immediate before ] (single line)
        line = re.sub(
            r',\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(\d+|\'[^\']*\'|\"[^\"]*\"|True|False)\s*\]',
            r", {'\1': \2} ]",
            line
        )
        # Second try: anywhere after comma (multi-line arrays)
        # This catches cases like: ..., \n        replaceMessage = 1
        # Only transform if line starts with whitespace (continuation) and has assignment
        if line.strip() and not line[0].isalpha() and '=' in line:
            # Look for pattern: leading whitespace, identifier = value
            # Can be followed by comma, ], or end of line
            line = re.sub(
                r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(\d+|True|False)(\s*[,\]]|\s*$)',
                r"\1{'\2': \3}\4",
                line
            )

        # Special case: Function(...) = value (expression can't be assigned to)
        # This happens with AnswerHints( Formula(...) => "msg", ... )
        # Transform to tuple pairs: (Formula(...), "msg")
        # Pattern: CapitalizedWord(...) = "string" or CapitalizedWord(...) = number
        # Wrap in parens to make it a tuple element
        # String pattern handles both double and single quotes with any content
        # IMPORTANT: Use negative lookbehind (?<![a-z]) to ensure we don't match
        # capital letters that are part of a camelCase method name like withPostFilter
        line = re.sub(
            r'(?<![a-z])([A-Z][a-zA-Z0-9_]*\([^)]*\))\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')',
            r'(\1, \2)',
            line
        )

        # Special case: 'string' = value (string literal can't be keyword arg)
        # This happens with parserFunction('f(x,y)' => 'definition')
        # Also: "y'" = 'Real' in Context().variables.are() calls
        # Transform to comma-separated arguments: 'string', value
        # Pattern: quoted string = quoted value
        # Use more general pattern that handles quotes inside strings
        line = re.sub(
            r'"([^"]*?)"\s*=\s*(["\'][^"\']*["\'])',
            r'"\1", \2',
            line
        )
        line = re.sub(
            r"'([^']*?)'\s*=\s*([\"'][^\"']*[\"'])",
            r"'\1', \2",
            line
        )

        # Special case: identifier = { dict } in .add() calls (should be 'identifier', { dict })
        # This happens with Context().functions.add(name => { ... })
        # The => was converted to = by replace_hash_arrow, but should be string + comma
        # Pattern: .add( word = {
        # Convert to: .add( 'word', {
        # Only apply after .add( to avoid breaking legitimate keyword arguments
        line = re.sub(
            r'(\.add\(\s*)([a-z_][a-zA-Z0-9_]*)\s*=\s*\{',
            r"\1'\2', {",
            line
        )

        # Special case: Wrap [ ... ], [ ... ] pairs in parens for AnswerHints
        # After transforming ] => [ to ], [ we need to wrap in parens to make a tuple
        # This is ONLY for AnswerHints( [ arr1 ], [ arr2 ] ) patterns, not all functions
        # Other functions like random_coprime expect separate arguments, not a tuple
        if '], [' in line and 'AnswerHints' in line:
            # Wrap [ ... ], [ ... ] patterns in AnswerHints calls
            # Match: [ ... ], [ ... ] where arrays can span lines (use non-greedy)
            line = re.sub(
                r'(\(|\,)\s*(\[(?:[^\[\]]|\[[^\]]*\])*\])\s*,\s*(\[(?:[^\[\]]|\[[^\]]*\])*\])',
                r'\1 (\2, \3)',
                line
            )

        # Transform Perl string comparison operators (must be done carefully)
        # eq → == (string equality)
        # ne → != (string inequality)
        # lt → < (less than)
        # gt → > (greater than)
        # le → <= (less than or equal)
        # ge → >= (greater than or equal)
        # Use word boundaries to avoid matching inside identifiers
        # Use negative lookahead to avoid matching variable names (followed by =, ., (, [)
        # This prevents: $gt = ... from becoming > = ...
        line = re.sub(r'\beq\b(?!\s*[=\.\(\[])', '==', line)
        line = re.sub(r'\bne\b(?!\s*[=\.\(\[])', '!=', line)
        line = re.sub(r'\blt\b(?!\s*[=\.\(\[])', '<', line)
        line = re.sub(r'\bgt\b(?!\s*[=\.\(\[])', '>', line)
        line = re.sub(r'\ble\b(?!\s*[=\.\(\[])', '<=', line)
        line = re.sub(r'\bge\b(?!\s*[=\.\(\[])', '>=', line)

        # Transform Perl namespace separator: Package::Function → Package.Function
        line = re.sub(
            r'([a-zA-Z_][a-zA-Z0-9_]*)::([a-zA-Z_][a-zA-Z0-9_]*)', r'\1.\2', line)

        # Transform Perl string concatenation operator: ' . ' → ' + '
        # Only when surrounded by spaces or between string literals/variables
        # Match: 'str' . 'str' or var . 'str' or 'str' . var
        # Special handling for expressions: wrap non-string operands in str()
        # Case 1: When second operand is ( expression ), wrap in str()
        line = re.sub(
            r'(\)|\'|\"|\w)\s+\.\s+(\([^)]+\))', r'\1 + str(\2)', line)
        # Case 2: Regular concatenation with strings/variables
        line = re.sub(r'(\)|\'|\"|\w)\s+\.\s+(\'|\"|\w)', r'\1 + \2', line)

        # Also handle continuation lines starting with . (Perl string concat)
        # Match: ^\s+. "string" and convert to + "string"
        # If it's . (expr), wrap in str()
        line = re.sub(r'^(\s+)\.\s+(\([^)]+\))', r'\1+ str(\2)', line)
        # Otherwise just convert . to +
        line = re.sub(r'^(\s+)\.\s+', r'\1+ ', line)

        # Transform Perl string repetition operator: x → *
        # Match: 'str' x 3 or var x num
        # Use word boundaries to avoid matching variable named 'x'
        # Pattern: (value) x (number) where x is surrounded by spaces
        line = re.sub(r'(\)|\'|\"|\w)\s+x\s+(\d+|\w+)', r'\1 * \2', line)

        # Transform Perl regex literals: qr/pattern/flags → r"pattern"
        # Example: qr/[ty]'*/i → r"[ty]'*"
        # Common flags: i (case insensitive), m (multiline), s (single line)
        # For now, convert to raw string and ignore flags (most patterns don't need compilation)
        def convert_qr_regex(match):
            pattern = match.group(1)
            # Escape any double quotes in the pattern
            escaped_pattern = pattern.replace('"', '\\"')
            return f'r"{escaped_pattern}"'

        line = re.sub(r'qr/([^/]+)/\w*', convert_qr_regex, line)

        # Note: do-while/do-until loops are handled in main preprocess loop
        # to allow multi-line output

        # Transform Perl for-loops to Python for-in loops
        # Patterns:
        # 1. for VAR (EXPR) { → for VAR in EXPR:
        # 2. for my VAR (EXPR) { → for VAR in EXPR:  (remove 'my' keyword)
        # 3. Handle Perl range: (START .. END) → range(START, END+1)

        # Convert for-loop syntax FIRST (before range conversion)
        # This captures the expression and handles Perl range within it
        def convert_for_loop(match):
            var = match.group(1)
            expr = match.group(2)
            # Handle Perl range inside expression: START .. END → range(START, END+1)
            expr = re.sub(r'(\d+|[a-zA-Z_]\w*)\s*\.\.\s*(\d+|[a-zA-Z_]\w*)',
                          lambda m: f'range({m.group(1).strip()}, {m.group(2).strip()}+1)',
                          expr)
            return f'for {var} in {expr}:'

        # Pattern: for my? VAR (EXPR) { → for VAR in EXPR:
        line = re.sub(
            r'\bfor\s+my\s+([a-zA-Z_]\w*)\s*\(([^)]+)\)\s*\{', convert_for_loop, line)
        line = re.sub(
            r'\bfor\s+([a-zA-Z_]\w*)\s*\(([^)]+)\)\s*\{', convert_for_loop, line)

        # Remove Perl 'my' keyword from variable declarations
        # Pattern: my VAR = → VAR =
        # This handles inline statements like: for i: my x = i * 2; my y = x + 1
        line = re.sub(r'\bmy\s+([a-zA-Z_]\w*)\s*=', r'\1 =', line)

        # Clean up stray closing braces from converted for-loops and blocks
        # Pattern 1: statements; } at end of line → statements (remove trailing brace + semicolon)
        line = re.sub(r';\s*}$', '', line)
        # Pattern 2: ) } at end of line (function call followed by brace)
        line = re.sub(r'\)\s*}$', ')', line)

        # Transform Perl map with blocks: map { EXPR } LIST
        # map { random(1, 10) } 0 .. 7  →  [random(1, 10) for _ in range(0, 8)]
        # map { $f->eval(x => $_) } 0 .. 2  →  [f.eval(x=_) for _ in range(0, 3)]
        map_match = re.search(
            r'\bmap\s*\{\s*([^}]+)\}\s+(\d+)\s*\.\.\s*(\d+)', line)
        if map_match:
            expr = map_match.group(1).strip()
            # Fix: map { } blocks had => converted to : by replace_hash_arrow
            # Inside map blocks, => should be = (keyword args), not : (dict)
            # Convert : back to = for function arguments
            # Pattern: identifier : expression (but not inside nested strings)
            expr = re.sub(r'(\w+)\s*:\s*', r'\1=', expr)
            start = int(map_match.group(2))
            end = int(map_match.group(3))
            # Python range is exclusive on the right, Perl .. is inclusive
            replacement = f'[{expr} for _ in range({start}, {end}+1)]'
            line = line[:map_match.start()] + replacement + \
                line[map_match.end():]

        # Transform Perl range operator: START .. END → range(START, END+1)
        # But only if not already handled by map
        if '..' in line and 'range(' not in line:
            line = re.sub(r'(\d+)\s*\.\.\s*(\d+)',
                          lambda m: f'range({m.group(1)}, {int(m.group(2))+1})', line)

        # Transform Perl unless → if not
        line = re.sub(r'\bunless\s+', 'if not ', line)

        # Transform Perl ternary operator: condition ? true_value : false_value
        # → true_value if condition else false_value
        # Handle nested ternaries by converting innermost-to-outermost
        # Loop until no more ternaries found (max 10 iterations to prevent infinite loop)
        max_iterations = 10
        iteration = 0
        while '?' in line and ':' in line and iteration < max_iterations:
            iteration += 1
            old_line = line

            # Find the LAST ? (innermost in nested ternaries)
            # Track parenthesis depth to find matching :
            question_pos = -1
            for i in range(len(line) - 1, -1, -1):
                if line[i] == '?':
                    # Check if this is inside a string
                    in_string = False
                    for j in range(i):
                        if line[j] in ('"', "'") and (j == 0 or line[j-1] != '\\'):
                            in_string = not in_string
                    if not in_string:
                        question_pos = i
                        break

            if question_pos < 0:
                break

            # Find matching : after the ?
            depth = 0
            colon_pos = -1
            for i in range(question_pos + 1, len(line)):
                ch = line[i]
                if ch in '([{':
                    depth += 1
                elif ch in ')]}':
                    depth -= 1
                elif ch == ':' and depth == 0:
                    # Check if this looks like a ternary colon (not a dict key)
                    if i > 0 and not (line[i-1].isalnum() or line[i-1] == '_'):
                        colon_pos = i
                        break

            if colon_pos < 0:
                break

            # Find start of condition (work backwards from ?)
            expr_start = 0
            depth = 0
            for j in range(question_pos - 1, -1, -1):
                ch = line[j]
                if ch in ')]}':
                    depth += 1
                elif ch in '([{':
                    depth -= 1
                    if depth < 0:
                        expr_start = j + 1
                        break
                elif depth == 0 and ch in '=,(':
                    expr_start = j + 1
                    break

            # Find end of false value (work forwards from :)
            expr_end = len(line)
            depth = 0
            for j in range(colon_pos + 1, len(line)):
                ch = line[j]
                if ch in '([{':
                    depth += 1
                elif ch in ')]}':
                    depth -= 1
                    if depth < 0:
                        expr_end = j
                        break
                elif depth == 0 and ch in ',)':
                    expr_end = j
                    break

            # Extract components
            condition = line[expr_start:question_pos].strip()
            true_value = line[question_pos+1:colon_pos].strip()
            false_value = line[colon_pos+1:expr_end].strip()

            # Reconstruct with Python ternary
            before = line[:expr_start]
            after = line[expr_end:]
            line = f"{before}{true_value} if {condition} else {false_value}{after}"

            # If line didn't change, break to avoid infinite loop
            if line == old_line:
                break

        # Transform Perl statement modifiers: STATEMENT if/unless CONDITION
        # statement if condition → if condition: statement
        # statement unless condition → if not condition: statement
        # But be careful not to transform regular if/elsif/else blocks, comment lines, or Python ternaries
        if re.search(r'\S+.*\s+(if|unless)\s+\S+', line) and not re.match(r'^\s*(if|elsif|else|unless|#)', line):
            # Skip if this is a Python ternary (has 'if' with 'else' after it)
            # Python ternary: VALUE if CONDITION else OTHER_VALUE
            if ' if ' in line and ' else ' in line:
                # Check if 'else' comes after 'if' (Python ternary pattern)
                if_pos = line.find(' if ')
                else_pos = line.find(' else ')
                if if_pos >= 0 and else_pos > if_pos:
                    # This is a Python ternary, don't transform
                    pass
                else:
                    # Not a ternary, continue with statement modifier
                    # Check if this is a statement modifier (not a block if)
                    # Statement modifiers don't have colons or blocks after them
                    match = re.search(
                        r'^(\s*)(.+?)\s+(if|unless)\s+(.+)$', line)
                    # No block in statement
                    if match and '{' not in match.group(2):
                        indent = match.group(1)
                        statement = match.group(2).strip()
                        modifier = match.group(3)
                        condition = match.group(4).strip()

                        if modifier == 'if':
                            line = f"{indent}if {condition}: {statement}"
                        else:  # unless
                            line = f"{indent}if not ({condition}): {statement}"
            else:
                # No 'else', so can't be a Python ternary
                # Check if this is a statement modifier (not a block if)
                # Statement modifiers don't have colons or blocks after them
                match = re.search(r'^(\s*)(.+?)\s+(if|unless)\s+(.+)$', line)
                # No block in statement
                if match and '{' not in match.group(2):
                    indent = match.group(1)
                    statement = match.group(2).strip()
                    modifier = match.group(3)
                    condition = match.group(4).strip()

                    if modifier == 'if':
                        line = f"{indent}if {condition}: {statement}"
                    else:  # unless
                        line = f"{indent}if not ({condition}): {statement}"

        # Transform Perl fat comma (hash key-value): key => value
        # Context-aware conversion:
        # - Inside { ... }: key => value → key: value (Python dict)
        # - In function args: key => value → key = value (named parameter)
        # Avoid array refs: ] => [

        if '] =>' not in line and '} =>' not in line:
            # Strategy: Find all occurrences of => and determine context
            # by checking if we're inside curly braces
            result = []
            i = 0
            brace_depth = 0
            paren_depth = 0

            while i < len(line):
                ch = line[i]

                # Track brace/paren depth
                if ch == '{':
                    brace_depth += 1
                elif ch == '}':
                    brace_depth -= 1
                elif ch == '(':
                    paren_depth += 1
                elif ch == ')':
                    paren_depth -= 1

                # Check for =>
                if i < len(line) - 1 and line[i:i+2] == '=>':
                    # Decide what to replace with based on context
                    if brace_depth > 0:
                        # Inside braces: use colon for dict
                        result.append(':')
                    else:
                        # Outside braces (function params): use equals
                        result.append(' =')
                    i += 2  # Skip both characters
                else:
                    result.append(ch)
                    i += 1

            line = ''.join(result)

        # Quote unquoted dictionary keys: { key: value } or { key : value } → { 'key': value }
        # Match word keys followed by optional whitespace and colon
        def quote_dict_key(match):
            key = match.group(1)
            spaces = match.group(2)  # Preserve whitespace before colon
            # Check if key is already quoted or is a Python keyword/builtin
            if key in ['True', 'False', 'None']:
                return f'{key}{spaces}:'
            return f"'{key}'{spaces}:"

        line = re.sub(r'(?<=[{,\s])(\w+)(\s*):', quote_dict_key, line)

        # Remove trailing semicolons (optional in Python)
        line = re.sub(r';\s*$', '', line)

        # Pass through - Python handles # comments same as Perl/PG
        return line

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

        # Split content into segments: text, $var, \{...}
        segments = []
        pos = 0

        while pos < len(content):
            # Look for $var or \{...}
            var_match = re.search(r'\$([a-zA-Z_][a-zA-Z0-9_]*)', content[pos:])
            func_match = re.search(r'\\{([^}]+)\\}', content[pos:])

            # Find which comes first
            next_var_pos = pos + var_match.start() if var_match else len(content)
            next_func_pos = pos + func_match.start() if func_match else len(content)

            if next_var_pos < next_func_pos:
                # Variable comes first
                # Add text before variable
                if next_var_pos > pos:
                    text_segment = content[pos:next_var_pos]
                    segments.append(repr(text_segment))

                # Add variable
                var_name = var_match.group(1)
                # Check if it's a known macro function
                if var_name in ('PAR', 'BR', 'BBOLD', 'EBOLD', 'BITALIC', 'EITALIC', 'BCENTER', 'ECENTER', 'BUL', 'EUL'):
                    segments.append(f"{var_name}()")
                else:
                    segments.append(f"str({var_name})")

                pos = next_var_pos + len(var_match.group(0))

            elif next_func_pos < len(content):
                # Function call comes first
                # Add text before function
                if next_func_pos > pos:
                    text_segment = content[pos:next_func_pos]
                    segments.append(repr(text_segment))

                # Add function call - transform Perl syntax to Python
                func_code = func_match.group(1).strip()
                # Convert $var->method() to var.method()
                func_code = self._transform_line(func_code)
                segments.append(func_code)

                pos = next_func_pos + len(func_match.group(0))

            else:
                # No more variables or functions - add remaining text
                if pos < len(content):
                    text_segment = content[pos:]
                    segments.append(repr(text_segment))
                break

        # Join segments with commas for TEXT() call
        if not segments:
            return '""'

        return ", ".join(segments)

    def _escape_triple_quotes(self, text: str) -> str:
        """Escape special characters in text for Python string literals."""
        # Escape backslashes first (before escaping quotes)
        text = text.replace("\\", "\\\\")
        # Then escape triple quotes
        text = text.replace("'''", r"\'\'\'")
        text = text.replace('"""', r'\"\"\"')
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
                # Found comment start - return everything before it
                return line[:i].rstrip()

        # No comment found
        return line.rstrip()

    def _transform_pgml_evaluators(self, pgml_content: str) -> str:
        """
        Transform Perl syntax to Python in PGML evaluator expressions.

        PGML allows inline evaluators like: [_]{$answer->cmp()}
        We need to convert Perl method calls (->)  to Python (.).

        Args:
            pgml_content: PGML markup text

        Returns:
            Transformed PGML with Python syntax in evaluator expressions
        """
        import re

        # Pattern to match {$var->method(...)} or {$var.method(...)}
        # We want to convert -> to . inside {...} that comes after [_]
        # Use a more careful approach: find all {code} blocks and transform them

        result = []
        i = 0
        while i < len(pgml_content):
            # Look for { that might start an evaluator
            if pgml_content[i] == '{':
                # Find the matching closing brace
                brace_depth = 1
                j = i + 1
                while j < len(pgml_content) and brace_depth > 0:
                    if pgml_content[j] == '{':
                        brace_depth += 1
                    elif pgml_content[j] == '}':
                        brace_depth -= 1
                    j += 1

                if brace_depth == 0:
                    # Found matching closing brace
                    code_block = pgml_content[i+1:j-1]

                    # Transform Perl syntax to Python
                    # Convert -> to .
                    transformed = code_block.replace('->', '.')
                    # Remove Perl $ sigils from variables (keep $ in $$ for LaTeX)
                    transformed = re.sub(
                        r'\$([a-zA-Z_]\w*)', r'\1', transformed)

                    result.append('{')
                    result.append(transformed)
                    result.append('}')
                    i = j
                    continue

            result.append(pgml_content[i])
            i += 1

        return ''.join(result)

    def _transform_load_macros(self, macro_list_str: str) -> tuple[list[str], str]:
        """
        Transform loadMacros() call to Python imports.

        Args:
            macro_list_str: The content inside loadMacros(...), e.g., '"PG.pl", "PGML.pl"'

        Returns:
            Tuple of (import_lines, comment)

        Example:
            Input: '"PGstandard.pl", "MathObjects.pl", "PGML.pl"'
            Output: (['from pg.macros.core.pg_core import *', ...],
                    '# loadMacros("PGstandard.pl", "MathObjects.pl", "PGML.pl") - loaded')
        """
        import re

        # Extract quoted strings
        macros = re.findall(r'["\']([^"\']+)["\']', macro_list_str)

        # Mapping of .pl files to Python imports
        macro_imports = {
            "PG.pl": "from pg.macros.core.pg_core import DOCUMENT, TEXT, ANS, ENDDOCUMENT, SOLUTION, HINT",
            "PGstandard.pl": "from pg.macros.answers.pg_answer_macros import num_cmp, str_cmp, fun_cmp",
            "PGbasicmacros.pl": "from pg.macros.core.pg_basic_macros import ans_rule, beginproblem, PAR",
            "MathObjects.pl": "from pg.math import Context, Real, Complex, Formula, Interval",
            "PGML.pl": "from pg.pgml import PGML",
            "contextFraction.pl": "from pg.math import Fraction",
            "PGcourse.pl": "# PGcourse.pl - course-specific (skipped)",
        }

        # Generate import lines
        import_lines = []
        loaded_macros = []

        for macro in macros:
            if macro in macro_imports:
                import_line = macro_imports[macro]
                if not import_line.startswith("#"):
                    import_lines.append(import_line)
                loaded_macros.append(macro)

        # Create comment showing what was loaded
        comment = f'# loadMacros({", ".join(repr(m) for m in loaded_macros)}) - loaded' if loaded_macros else "# loadMacros() - no recognized macros"

        return (import_lines, comment)


def convert_pg_file(
    source_path: str | Path,
    *,
    output_path: str | Path | None = None,
    use_sandbox_macros: bool = False,
    overwrite: bool = False,
    encoding: str = "utf-8",
    preprocessor: PGPreprocessor | None = None,
) -> tuple[Path, PreprocessResult]:
    """Convert a .pg file to Python and persist the result as .pyg."""
    pg_path = Path(source_path)
    if not pg_path.exists():
        raise FileNotFoundError(f"PG source file not found: {pg_path}")

    processor = preprocessor or PGPreprocessor()
    pg_source = pg_path.read_text(encoding=encoding)
    result = processor.preprocess(
        pg_source, use_sandbox_macros=use_sandbox_macros)

    output = Path(output_path) if output_path else pg_path.with_suffix('.pyg')
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.code, encoding=encoding)
    return output, result
