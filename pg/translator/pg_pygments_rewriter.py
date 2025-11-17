"""
Pygments-based fallback rewriter for PG source code.

This module provides token-level code transformation when the Lark
parser cannot handle certain Perl constructs. It uses Pygments to
tokenize the code and applies conservative transformations.
"""

from __future__ import annotations

import re
from typing import List
from pygments.lexers import get_lexer_by_name
from pygments.token import Token


class PGPygmentsRewriter:
    """
    Fallback rewriter using Pygments for conservative token replacement.

    This class handles transformation of Perl code to Python when the
    Lark parser fails. It operates at the token level using Pygments,
    applying transformations like:
    - Sigil removal ($var -> var)
    - Arrow operator (-> becomes .)
    - Namespace operator (:: becomes .)
    - Fat comma (=> becomes : or = depending on context)
    - Hash/array subscripting
    - String interpolation
    """

    def __init__(self):
        """Initialize the Pygments rewriter with a Perl lexer."""
        self._perl_lexer = get_lexer_by_name("perl")

    def desigil(self, name: str) -> str:
        """
        Remove Perl sigils from variable names.

        Converts:
            $var -> var
            @arr -> arr
            %hash -> hash

        Args:
            name: Variable name with sigil

        Returns:
            Variable name without sigil
        """
        if name and name[0] in '$@%':
            return name[1:]
        return name

    def convert_regexes(self, code: str) -> str:
        """
        Convert Perl regex literals to Python re.compile() calls.

        Converts:
            qr/pattern/flags -> re.compile(r"pattern", flags)

        Args:
            code: Code containing Perl regex literals

        Returns:
            Code with Python regex compile calls
        """
        def replace_regex(match):
            pattern = match.group(1)
            flags_str = match.group(2) if match.group(2) else ''

            # Convert Perl regex flags to Python re flags
            flag_map = {
                'i': 're.IGNORECASE',
                'm': 're.MULTILINE',
                's': 're.DOTALL',
                'x': 're.VERBOSE',
            }
            py_flags = [flag_map[f] for f in flags_str if f in flag_map]

            if py_flags:
                flags_part = ' | '.join(py_flags)
                return f're.compile(r"{pattern}", {flags_part})'
            else:
                return f're.compile(r"{pattern}")'

        return re.sub(r'qr/([^/]*)/([imsxo]*)', replace_regex, code)

    def rewrite_with_pygments(self, code: str) -> str:
        """
        Fallback rewrite using Pygments for conservative token replacement.

        This method tokenizes the code using Pygments and applies a series
        of transformations to convert Perl syntax to Python. It handles:
        - Variable sigils
        - Method arrows (->)
        - Namespace operators (::)
        - Hash/array subscripting
        - String interpolation
        - Fat comma (=>)

        Args:
            code: Perl-like code to transform

        Returns:
            Python code
        """
        # First convert regex literals
        code = self.convert_regexes(code)

        tokens = list(self._perl_lexer.get_tokens_unprocessed(code))
        result: List[str] = []
        i = 0

        # Track context for proper transformation
        brace_context_stack: List[bool] = []  # True if hash literal
        bracket_depth = 0

        while i < len(tokens):
            _, ttype, text = tokens[i]

            # Preserve comments verbatim
            if ttype in Token.Comment:
                result.append(text)
                i += 1
                continue

            # Handle string interpolation
            if ttype in Token.Literal.String and text.startswith('"') and '$' in text:
                content = text[1:-1] if len(text) >= 2 else text
                escaped = content.replace('\\', '\\\\')
                escaped = escaped.replace('{', '{{').replace('}', '}}')
                converted = re.sub(r'\$([a-zA-Z_][a-zA-Z0-9_]*)', r'{\1}', escaped)
                result.append(f'f"{converted}"')
                i += 1
                continue

            # Handle variables
            if ttype in Token.Name.Variable:
                # Special handling for $# (array last index)
                if text == '$#' or text.startswith('$#'):
                    j = i + 1
                    while j < len(tokens) and tokens[j][1] in Token.Text.Whitespace:
                        j += 1
                    if j < len(tokens) and tokens[j][1] in Token.Name.Variable:
                        array_name = self.desigil(tokens[j][2])
                        result.append(f"len({array_name}) - 1")
                        i = j + 1
                        continue

                result.append(self.desigil(text))
                i += 1
                continue

            # Handle namespace tokens
            if ttype in Token.Name.Namespace:
                result.append(text.replace('::', '.'))
                i += 1
                continue

            # Handle hash subscripting: $h{key} -> h['key']
            if text == '{' and i > 0:
                prev_token = tokens[i-1]
                if (prev_token[1] in Token.Name.Variable or
                    prev_token[2] in (')', '}', '>')):
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
                    if inner_text and inner_text[0] not in "'\"":
                        inner_text = f"'{inner_text}'"
                    result.append('[' + inner_text + ']')
                    i = j + 1
                    continue

            # Handle operators
            if ttype in Token.Operator or ttype in Token.Punctuation:
                # Method arrow: -> becomes .
                if text == '-' and i + 1 < len(tokens) and tokens[i+1][2] == '>':
                    i += 2
                    while i < len(tokens) and tokens[i][2] == '':
                        i += 1

                    if i < len(tokens):
                        next_text = tokens[i][2]
                        if next_text != '{':  # Not hash subscript
                            result.append('.')
                            # Check if it's a method call or property access
                            if tokens[i][1] in Token.Name:
                                result.append(next_text)
                                # Look ahead for parentheses
                                j = i + 1
                                while j < len(tokens) and tokens[j][2] == '':
                                    j += 1
                                has_parens = (j < len(tokens) and tokens[j][2] == '(')

                                # Known properties that don't need ()
                                property_names = {
                                    "transpose", "inverse", "norm", "dimensions",
                                    "trace", "det", "determinant", "reduce", "value"
                                }
                                if not has_parens and next_text not in property_names:
                                    result.append('()')
                                i += 1
                                continue
                    else:
                        result.append('.')
                    continue

                # Namespace separator: :: becomes .
                if text == ':' and i + 1 < len(tokens) and tokens[i+1][2] == ':':
                    result.append('.')
                    i += 2
                    continue

                # Fat comma: =>
                if text == '=' and i + 1 < len(tokens) and tokens[i+1][2] == '>':
                    if brace_context_stack and brace_context_stack[-1]:
                        result.append(': ')
                    else:
                        result.append(' = ')
                    i += 2
                    continue

                # Concatenation operator: . becomes +
                if text == '.':
                    # Check context to distinguish from method access
                    is_concat = False
                    if i > 0 and i + 1 < len(tokens):
                        j = i - 1
                        while j >= 0 and tokens[j][1] in Token.Text.Whitespace:
                            j -= 1
                        k = i + 1
                        while k < len(tokens) and tokens[k][1] in Token.Text.Whitespace:
                            k += 1

                        if j >= 0 and k < len(tokens):
                            # Check if both sides look like expressions
                            prev_text = tokens[j][2]
                            next_text = tokens[k][2]
                            is_expr_before = (
                                tokens[j][1] in Token.Name.Variable or
                                prev_text in (')', ']', '}') or
                                tokens[j][1] in Token.Literal.String
                            )
                            is_expr_after = (
                                tokens[k][1] in Token.Name.Variable or
                                next_text in ('(', '[', '{') or
                                tokens[k][1] in Token.Literal.String or
                                tokens[k][1] in Token.Name
                            )
                            # If both sides look like expressions, it's concatenation
                            if is_expr_before and is_expr_after:
                                is_concat = True

                    if is_concat:
                        result.append(' + ')
                        i += 1
                        continue

            # Track brace context for hash literals
            if text == '{':
                is_hash_literal = False
                if i > 0:
                    j = i - 1
                    while j >= 0 and tokens[j][1] in Token.Text.Whitespace:
                        j -= 1
                    if j >= 0:
                        prev_token = tokens[j][2]
                        if prev_token in ('=>', '>', '(', '[', ',', '='):
                            is_hash_literal = True
                else:
                    is_hash_literal = True
                brace_context_stack.append(is_hash_literal)
            elif text == '}':
                if brace_context_stack:
                    brace_context_stack.pop()

            # Track bracket depth
            if text == '[':
                bracket_depth += 1
            elif text == ']':
                bracket_depth = max(0, bracket_depth - 1)

            # Drop trailing semicolon at end
            if text == ';' and i == len(tokens) - 1:
                i += 1
                continue

            result.append(text)
            i += 1

        rewritten = ''.join(result).rstrip()

        # Post-processing transformations
        # Convert .with( to .with_params(
        rewritten = re.sub(r'\.with\(', '.with_params(', rewritten)

        # Convert .reduce() to .reduce (property in Python MathObjects)
        rewritten = re.sub(r'\.reduce\(\)', '.reduce', rewritten)

        # Convert range operators
        rewritten = re.sub(
            r'(\d+)\s*\.\.\s*(len\([^)]*\)\s*-\s*1)',
            r'range(\1, \2 + 1)',
            rewritten
        )
        rewritten = re.sub(
            r'(\d+)\s*\.\.\s*(\d+)',
            r'range(\1, \2 + 1)',
            rewritten
        )

        return rewritten
