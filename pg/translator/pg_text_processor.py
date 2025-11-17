"""
Text preprocessing utilities for PG source files.

This module handles various text transformations needed before parsing,
including heredoc conversion, reference dereference fixes, and string
interpolation.
"""

from __future__ import annotations

import re
from typing import List


class PGTextProcessor:
    """
    Handles text preprocessing transformations for PG source code.

    This class provides utilities for converting Perl-specific text
    constructs (heredocs, string interpolation, reference dereferences)
    into forms that are easier to parse and convert to Python.
    """

    def fix_reference_dereferences(self, pg_source: str) -> str:
        """
        Fix Perl reference dereference arrows for parsing.

        Converts:
            $x->[$i] to $x[$i]
            $x->{key} to $x{key}
            $#$x to len(x) - 1
            @$x to list($x)

        This must happen before parsing because the grammar doesn't have
        rules for ->[ or ->{ patterns, and it doesn't understand $# for
        array length.

        Args:
            pg_source: Source code to transform

        Returns:
            Transformed source code
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

    def convert_heredocs_global(self, pg_source: str) -> str:
        """
        Convert Perl heredocs (<<END_MARKER) and qq/.../ to Python strings.

        Processes the entire source before line splitting to handle heredocs
        properly.

        Converts:
            HEADER_TEXT(MODES(TeX => '', HTML => <<END_STYLE));
            <style>...</style>
            END_STYLE

        To:
            HEADER_TEXT(MODES(TeX => '', HTML => '''<style>...</style>'''))

        Also converts:
            $var = qq/content here/;

        To:
            $var = '''content here''';

        Args:
            pg_source: Source code to transform

        Returns:
            Transformed source code with heredocs converted to triple-quoted strings
        """
        lines = pg_source.split('\n')
        result_lines: List[str] = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check for qq/.../ or qq{...} (Perl quoted strings)
            qq_match = re.search(r'qq([/\{\[\(\|])', line)
            if qq_match:
                delimiter = qq_match.group(1)
                closing_delim = {
                    '[': ']', '{': '}', '(': ')', '/': '/', '|': '|'
                }.get(delimiter, delimiter)

                before = line[:qq_match.start()]
                after_qq_start = line[qq_match.end():]

                content_lines: List[str] = []
                remaining = after_qq_start

                # Check if closing delimiter is on the same line
                close_idx = remaining.find(closing_delim)
                if close_idx != -1:
                    content = remaining[:close_idx]
                    after_content = remaining[close_idx + 1:]

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
                            content_lines.append(current[:close_idx])
                            after_content = current[close_idx + 1:]
                            break
                        else:
                            content_lines.append(current)
                        i += 1

                    content = '\n'.join(content_lines)
                    content = content.replace('\\', '\\\\')
                    content = content.replace("'''", "\\'''")

                    new_line = f"{before}'''{content}'''{after_content}"
                    result_lines.append(new_line)

            # Check for traditional heredoc (<<MARKER)
            elif re.search(r'<<([A-Z_][A-Z0-9_]*)', line):
                heredoc_match = re.search(r'<<([A-Z_][A-Z0-9_]*)', line)
                marker = heredoc_match.group(1)

                before = line[:heredoc_match.start()]
                after_marker = line[heredoc_match.end():]

                # Collect content until we find the marker on its own line
                content_lines: List[str] = []
                i += 1
                while i < len(lines):
                    current = lines[i]
                    if re.match(rf'^\s*{re.escape(marker)}\s*$', current):
                        break
                    content_lines.append(current)
                    i += 1

                content = '\n'.join(content_lines)
                content = content.replace('\\', '\\\\')
                content = content.replace("'''", "\\'''")

                new_line = f"{before}'''{content}'''{after_marker}"
                result_lines.append(new_line)
            else:
                result_lines.append(line)

            i += 1

        return '\n'.join(result_lines)

    def convert_string_interpolation(self, line: str) -> str:
        """
        Convert Perl string interpolation to Python f-strings.

        Converts:
            "The answer is $x"

        To:
            f"The answer is {x}"

        Args:
            line: Line of code to transform

        Returns:
            Transformed line with f-strings
        """
        def repl(match: re.Match[str]) -> str:
            quote = match.group(1)
            content = match.group(2)
            if quote == '"' and '$' in content:
                # Escape existing braces
                escaped = content.replace('{', '{{').replace('}', '}}')
                # Convert $var to {var}
                converted = re.sub(
                    r'\$([a-zA-Z_][a-zA-Z0-9_]*)', r'{\1}', escaped
                )
                return f'f"{converted}"'
            return match.group(0)

        return re.sub(r'(["\'])((?:[^\\]|\\.)*?)\1', repl, line)

    def strip_inline_comment(self, line: str) -> str:
        """
        Strip inline comments from a line, preserving strings.

        Args:
            line: Line of code potentially containing inline comments

        Returns:
            Line with inline comments removed
        """
        # Simple implementation: find # outside of strings
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
                return line[:i].rstrip()

        return line

    def transform_text_block(self, content: str) -> str:
        """
        Transform TEXT block content to Python expression(s).

        Handles:
        - Variable interpolation: $a → ", a, "
        - Function calls: \\{ ans_rule(20) \\} → ", ans_rule(20), "
        - LaTeX math: \\( ... \\) → keep as-is
        - Special vars: $PAR → ", PAR(), "

        Args:
            content: Raw TEXT block content

        Returns:
            Python expression string suitable for TEXT() call
        """
        segments: List[str] = []
        pos = 0

        # Special variable markers that should be converted to function calls
        SPECIAL_VARS = {
            'PAR', 'BR', 'BBOLD', 'EBOLD', 'BITALIC', 'EITALIC',
            'BCENTER', 'ECENTER', 'BUL', 'EUL'
        }

        while pos < len(content):
            var_match = re.search(r'\$([a-zA-Z_][a-zA-Z0-9_]*)', content[pos:])
            func_match = re.search(r'\\{([^}]+)\\}', content[pos:])

            next_var_pos = pos + var_match.start() if var_match else len(content)
            next_func_pos = pos + func_match.start() if func_match else len(content)

            if next_var_pos < next_func_pos:
                # Variable interpolation comes first
                if next_var_pos > pos:
                    text_segment = content[pos:next_var_pos]
                    segments.append(repr(text_segment))

                var_name = var_match.group(1)
                if var_name in SPECIAL_VARS:
                    segments.append(f"{var_name}()")
                else:
                    segments.append(f"str({var_name})")

                pos = next_var_pos + len(var_match.group(0))

            elif next_func_pos < len(content):
                # Function call comes first
                if next_func_pos > pos:
                    text_segment = content[pos:next_func_pos]
                    segments.append(repr(text_segment))

                func_code = func_match.group(1).strip()
                # Note: func_code transformation would need access to _compile_expr
                # For now, we'll just include it as a string placeholder
                segments.append(f"str({func_code})")

                pos = next_func_pos + len(func_match.group(0))
            else:
                # No more variables or functions
                if pos < len(content):
                    text_segment = content[pos:]
                    segments.append(repr(text_segment))
                break

        if not segments:
            return '""'

        return ', '.join(segments)
