"""Render PGML markup to HTML."""

import re
from typing import Dict, Any, Tuple
import re


class PGMLRenderer:
    """Render PGML markup to HTML."""

    def __init__(self, variables: Dict[str, Any]):
        self.variables = variables
        self.answer_counter = 0
        # answer_id → either string correct value or a dict with metadata
        self.answer_blanks: Dict[str, Any] = {}
        # Track MultiAnswer indices
        self._multi_indices: Dict[str, int] = {}

    def render(self, pgml: str) -> Tuple[str, Dict[str, str]]:
        """
        Render PGML to HTML.

        Returns:
            (html_string, answer_blanks_dict)
        """
        html = pgml

        # 1. Variable interpolation FIRST (before any bracket/brace processing)
        # This prevents variables like [$a] from being corrupted by table simplification
        # Match both [$varname] and [varname] (preprocessor may have removed $)
        # Must start with a letter (not underscore) to avoid matching [_] answer blanks
        html = re.sub(r'\[\$?([a-zA-Z]\w*)\]', self._interpolate_var, html)

        # 1.5. Convert LaTeX-style math delimiters to Markdown/KaTeX format
        # Some PGML content uses \(...\) and \[...\] instead of [` ... `]
        # Convert inline math \(...\) to $...$
        html = re.sub(r'\\\((.*?)\\\)', r'$\1$', html, flags=re.DOTALL)
        # Convert display math \[...\] to $$...$$
        html = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', html, flags=re.DOTALL)

        # 2. Remove PGML table constructs (simplify for MVP)
        # These are advanced layout features: [# ... #] and [. ... .]
        html = self._simplify_tables(html)

        # 3. Display math: [`` ... ``] → KaTeX display math
        html = re.sub(r'\[``(.*?)``\]', r'$$\1$$', html, flags=re.DOTALL)

        # 4. Inline math: [` ... `] → KaTeX inline math
        html = re.sub(r'\[`(.*?)`\]', r'$\1$', html)

        # 5. Answer blanks: [_____]{$answer} or [_]{$answer}
        # Also handle optional width specifier: [_]{$answer}{15}
        html = re.sub(r'\[_+\]\{([^}]+)\}(?:\{[0-9]+\})?',
                      self._create_answer_blank, html)

        # 5.5 Variable interpolation in LaTeX math contexts AFTER blanks are handled
        # This prevents $answer in cmp chains from being expanded prematurely
        html = self._interpolate_variables_in_math(html)

        # 6. Formatting → Markdown
        # Bold: [*text*] → **text**
        html = re.sub(r'\[\*(.*?)\*\]', r'**\1**', html)
        # Italic: [|text|] → *text*
        html = re.sub(r'\[\|(.*?)\|\]', r'*\1*', html)
        # Underline: [_text_] → __text__ (approximation)
        html = re.sub(r'\[_(.*?)_\]', r'__\1__', html)

        # 7. Lists (already Markdown with leading *) - nothing to do here

        # 8. Cleanup: Remove any remaining PGML artifacts (conservative)
        # IMPORTANT: Do not touch LaTeX curly braces or math content
        # Only remove trailing PGML table options like "]*{ ... }"
        html = re.sub(r"\]\s*\*\s*\{[^}]+\}", "]", html)

        # 9. Paragraphs: ensure double newlines between blocks (Markdown)
        # Normalize Windows newlines and collapse extra spaces
        html = html.replace('\r\n', '\n')
        # Ensure we have a trailing newline
        if not html.endswith('\n'):
            html += '\n'

        return html, self.answer_blanks

    def _simplify_tables(self, pgml: str) -> str:
        """
        Simplify PGML table constructs for MVP.

        PGML tables use [# ... #] for rows and [. ... .] for cells.
        For MVP, we extract the content and ignore the layout directives.
        """
        # Remove table options like ]*{ padding => [...] }
        # Need to handle nested braces and brackets properly
        # Match: ]* followed by { then content (including nested [] and {}) then }
        def remove_table_options(text):
            # Use a more careful approach to handle nested structures
            result = []
            i = 0
            while i < len(text):
                # Look for ]*{
                if i < len(text) - 2 and text[i:i+3] == ']*{':
                    # Find the matching }
                    brace_count = 1
                    bracket_depth = 0
                    j = i + 3
                    while j < len(text) and brace_count > 0:
                        if text[j] == '[':
                            bracket_depth += 1
                        elif text[j] == ']' and bracket_depth > 0:
                            bracket_depth -= 1
                        elif text[j] == '{' and bracket_depth == 0:
                            brace_count += 1
                        elif text[j] == '}' and bracket_depth == 0:
                            brace_count -= 1
                        j += 1
                    # Replace ]*{...} with just ]
                    result.append(']')
                    i = j
                else:
                    result.append(text[i])
                    i += 1
            return ''.join(result)

        pgml = remove_table_options(pgml)

        # Convert [# ... #] table rows to simple line breaks
        pgml = re.sub(r'\[#\s*', '', pgml)
        pgml = re.sub(r'\s*#\]', '\n', pgml)

        # Convert [. ... .] table cells to simple spaces
        pgml = re.sub(r'\[\.\s*', '', pgml)
        pgml = re.sub(r'\s*\.\]', ' ', pgml)

        return pgml

    def _interpolate_var(self, match: re.Match) -> str:
        """Replace [$var] with variable value."""
        var_name = match.group(1)

        # Check if variable exists
        if var_name not in self.variables:
            # Variable not found - return a placeholder or empty string
            # to avoid showing raw $varname
            return f'[Variable ${var_name} not found]'

        value = self.variables.get(var_name)

        # Format numbers nicely
        if isinstance(value, float):
            # Remove trailing zeros
            return f'{value:g}'
        return str(value)

    def _create_answer_blank(self, match: re.Match) -> str:
        """Create HTML input for answer blank."""
        answer_expr = match.group(1)

        # Generate unique answer ID
        self.answer_counter += 1
        answer_id = f'AnSwEr{self.answer_counter:04d}'

        # Evaluate answer expression to get correct value or spec dict
        correct_value = self._eval_answer(answer_expr)
        
        # Handle None (variable not found or evaluation failed)
        if correct_value is None:
            # Variable not found - skip this answer blank
            # Return placeholder but don't register (will show as "No answer blanks detected")
            return f'___ANSWER_BLANK_{answer_id}___'
        
        # Store the evaluator object, dict spec, or string
        # Don't convert evaluator objects to strings!
        if isinstance(correct_value, dict):
            self.answer_blanks[answer_id] = correct_value
        elif hasattr(correct_value, 'cmp') or hasattr(correct_value, 'evaluate') or hasattr(correct_value, 'check'):
            # It's an evaluator object - wrap in dict format for consistency
            # This ensures it's properly recognized by the answer extraction system
            self.answer_blanks[answer_id] = {"evaluator": correct_value}
        else:
            # It's a simple value - convert to string
            self.answer_blanks[answer_id] = str(correct_value)

        # Return a placeholder that won't break markdown
        # The frontend will replace these with actual input fields
        return f'___ANSWER_BLANK_{answer_id}___'

    def _eval_answer(self, expr: str) -> Any:
        """
        Evaluate answer expression.

        The expression can be:
        - A simple variable: $answer or answer (preprocessor removes $)
        - A Compute() expression: Compute("x >= $a")
        - A literal string: "x >= 4"
        """
        expr = expr.strip()

        # Check if it's a simple variable name (preprocessor may have removed $)
        # Try to get it from variables first
        if expr.isidentifier() and expr in self.variables:
            result = self.variables[expr]
            # If it's an evaluator object, return it directly
            if hasattr(result, 'evaluate') or hasattr(result, 'cmp') or hasattr(result, 'check'):
                return result

        # Handle Python-style method calls: var.cmp(...) (preprocessor converts $var->cmp(...) to this)
        python_cmp_match = re.match(
            r'^(\w+)\s*\.\s*cmp\s*\((.*?)\)\s*(?:\.\w+\(.*?\))*$', expr, re.DOTALL)
        if python_cmp_match:
            var_name = python_cmp_match.group(1)
            options_str = python_cmp_match.group(2)
            base_val = self.variables.get(var_name, None)
            # Determine checker/options
            custom_checker_src, options = self._extract_custom_checker(
                options_str)

            if options.get('upToConstant', False):
                # Additive constant parity (antiderivative style)
                checker = 'up_to_additive_constant'
            else:
                checker = 'standard'
            if custom_checker_src is not None:
                checker = 'custom'
            # Build answer spec
            if hasattr(base_val, 'to_string'):
                value_str = base_val.to_string()
                variables = getattr(base_val, 'variables', [])
            else:
                value_str = str(base_val) if base_val is not None else expr
                variables = []
            spec = {
                'correct_value': value_str,
                'type': 'formula',
                'checker': checker,
                'variables': variables,
                'options': options,
                'evaluator': base_val,  # Store the actual evaluator object
            }
            if custom_checker_src is not None:
                spec['options']['custom_checker_src'] = custom_checker_src
            return spec

        # If it starts with $, it may be a variable or a method call like $var->cmp(...)
        if expr.startswith('$'):
            # Detect $var->cmp(options)
            # Allow method chaining after cmp, e.g., $ans->cmp(...)->withPostFilter(...)
            m = re.match(
                r'^\$(\w+)\s*->\s*cmp\s*\((.*?)\)\s*(?:->.*)?$', expr, re.DOTALL)
            if m:
                var_name = m.group(1)
                options_str = m.group(2)
                base_val = self.variables.get(var_name, None)
                # Determine checker/options
                custom_checker_src, options = self._extract_custom_checker(
                    options_str)
                checker = 'standard'
                if options.get('upToConstant', False):
                    # Additive constant parity (antiderivative style)
                    checker = 'up_to_additive_constant'
                if custom_checker_src is not None:
                    checker = 'custom'
                # Build answer spec
                if hasattr(base_val, 'to_string'):
                    value_str = base_val.to_string()
                    variables = getattr(base_val, 'variables', [])
                else:
                    value_str = str(base_val) if base_val is not None else expr
                    variables = []
                spec = {
                    'correct_value': value_str,
                    'type': 'formula',
                    'checker': checker,
                    'variables': variables,
                    'options': options,
                    'evaluator': base_val,  # Store the actual evaluator object
                }
                if custom_checker_src is not None:
                    spec['options']['custom_checker_src'] = custom_checker_src
                return spec

            # Detect $var->ans_rule(width)
            m2 = re.match(
                r'^\$(\w+)\s*->\s*ans_rule\s*\((.*?)\)\s*$', expr, re.DOTALL)
            if m2:
                group = m2.group(1)
                meta = self.variables.get(group, {})
                idx = self._multi_indices.get(group, 0)
                self._multi_indices[group] = idx + 1
                # Extract correct value if available in group meta
                correct_val = None
                variables = []
                atype = 'formula'
                ganswers = meta.get('answers') if isinstance(
                    meta, dict) else None
                if ganswers and idx < len(ganswers):
                    val = ganswers[idx]
                    if hasattr(val, 'to_string'):
                        correct_val = val.to_string()
                        variables = getattr(val, 'variables', [])
                        atype = 'formula'
                    else:
                        correct_val = str(val)
                        # Try to determine if numeric
                        try:
                            float(correct_val)
                            atype = 'number'
                        except Exception:
                            atype = 'formula'
                spec = {
                    'correct_value': correct_val if correct_val is not None else '',
                    'type': atype,
                    'checker': meta.get('checker', 'standard') if isinstance(meta, dict) else 'standard',
                    'variables': variables,
                    'options': meta.get('options', {}) if isinstance(meta, dict) else {},
                    'group': group,
                    'group_index': idx,
                }
                # If group has custom checker, mark it
                if isinstance(meta, dict) and meta.get('custom_checker_src'):
                    spec['checker'] = 'custom'
                    spec['options'] = dict(spec['options'])
                    spec['options']['custom_checker_src'] = meta['custom_checker_src']
                return spec

            # Simple variable reference $var
            var_name = expr.lstrip('$')
            result = self.variables.get(var_name, None)
            
            # If variable not found, return None (will be handled as error)
            if result is None:
                # Variable not found - this might be an error, but return None
                # The caller should handle this gracefully
                return None

            # If result is an evaluator object (has evaluate method), return it directly
            if hasattr(result, 'evaluate') or hasattr(result, 'cmp') or hasattr(result, 'check'):
                return result

            # MultiAnswer group variable: expand to per-blank spec
            if isinstance(result, dict) and result.get('__multi__'):
                group = var_name
                idx = self._multi_indices.get(group, 0)
                self._multi_indices[group] = idx + 1
                answers = result.get('answers', [])
                correct_val = ''
                variables = []
                atype = 'formula'
                if idx < len(answers):
                    val = answers[idx]
                    if hasattr(val, 'to_string'):
                        correct_val = val.to_string()
                        variables = getattr(val, 'variables', [])
                        atype = 'formula'
                    else:
                        correct_val = str(val)
                        try:
                            float(correct_val)
                            atype = 'number'
                        except Exception:
                            atype = 'formula'
                spec = {
                    'correct_value': correct_val,
                    'type': atype,
                    'checker': result.get('checker', 'standard'),
                    'variables': variables,
                    'options': result.get('options', {}),
                    'group': group,
                    'group_index': idx,
                }
                if result.get('custom_checker_src'):
                    spec['checker'] = 'custom'
                    spec['options'] = dict(spec['options'])
                    spec['options']['custom_checker_src'] = result['custom_checker_src']
                return spec
            # Interpolate variables if it's a string
            if isinstance(result, str):
                result = self._interpolate_variables_in_string(result)
            return result

        # Otherwise, it's a literal or expression - handle inline Compute/Formula -> cmp(...)
        inline = re.match(
            r"^(?:Compute|Formula)\(\s*['\"](.+?)['\"]\s*\)\s*->\s*cmp\s*\((.*?)\)\s*(?:->.*)?$", expr, flags=re.DOTALL)
        if inline:
            expr_str = inline.group(1)
            options_str = inline.group(2)
            custom_checker_src, options = self._extract_custom_checker(
                options_str)
            checker = 'standard'
            if options.get('upToConstant', False):
                checker = 'up_to_additive_constant'
            if custom_checker_src is not None:
                checker = 'custom'
            return {
                'correct_value': expr_str,
                'type': 'formula',
                'checker': checker,
                'variables': [],
                'options': options | ({'custom_checker_src': custom_checker_src} if custom_checker_src is not None else {}),
            }

        # Fallback: interpolate variables within the literal
        return self._interpolate_variables_in_string(expr)

    def _parse_cmp_options(self, s: str) -> Dict[str, Any]:
        """Parse a minimal subset of cmp(...) options from key=value or key=>value list.

        Supports both Python-style assignment (=) and Perl-style fat comma (=>).
        Only options that we currently use are parsed (e.g., upToConstant = 1).
        Unknown keys are ignored.
        """
        opts: Dict[str, Any] = {}
        # Split on commas not inside parentheses (cmp values here are simple)
        parts = [p.strip() for p in s.split(',') if p.strip()]
        for part in parts:
            # Handle both => (Perl) and = (Python) separators
            if '=>' in part:
                key, val = [x.strip() for x in part.split('=>', 1)]
            elif '=' in part:
                key, val = [x.strip() for x in part.split('=', 1)]
            else:
                continue
            # Strip surrounding quotes for key
            key = key.strip('"\'')
            # Normalize boolean/numeric values
            if val in ('1', 'true', 'True'):
                opts[key] = True
            elif val in ('0', 'false', 'False'):
                opts[key] = False
            else:
                # Best-effort int/float; otherwise raw string without quotes
                v_clean = val.strip('"\'')
                try:
                    opts[key] = int(v_clean)
                except ValueError:
                    try:
                        opts[key] = float(v_clean)
                    except ValueError:
                        opts[key] = v_clean
        return opts

    def _extract_custom_checker(self, s: str) -> Tuple[Any, Dict[str, Any]]:
        """Extract checker => sub { ... } from cmp options string if present.

        Returns (custom_checker_src or None, options_dict_without_checker).
        """
        # Find 'checker' => sub { ... }
        m = re.search(r"checker\s*=>\s*sub\s*\{", s)
        if not m:
            return None, self._parse_cmp_options(s)
        start = m.end() - 1  # position at '{'
        # Balance braces to find matching '}'
        depth = 0
        i = start
        while i < len(s):
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
            i += 1
        else:
            # Unbalanced; fall back
            return None, self._parse_cmp_options(s)
        code = s[start + 1: end]
        # Remove the checker segment from options string and parse the rest
        s_wo = s[: m.start()] + s[end + 1:]
        return code.strip(), self._parse_cmp_options(s_wo)

    def _interpolate_variables_in_string(self, text: str) -> str:
        """Replace $variable references in a string with their values."""
        def replacer(match):
            var_name = match.group(1)
            value = self.variables.get(var_name, f'${var_name}')
            # Format numbers nicely
            if isinstance(value, float):
                return f'{value:g}'
            return str(value)

        return re.sub(r'\$(\w+)', replacer, text)

    def _interpolate_variables_in_math(self, text: str) -> str:
        """Replace $variable references only inside LaTeX math regions ($...$ or $$...$$)."""
        def var_replacer(m):
            var_name = m.group(1)
            if var_name in self.variables:
                value = self.variables[var_name]
                if isinstance(value, float):
                    return f'{value:g}'
                elif hasattr(value, 'to_string'):
                    return value.to_string()
                else:
                    return str(value)
            return f'${var_name}'

        def replace_in(content: str) -> str:
            return re.sub(r'\$(\w+)', var_replacer, content)

        # Replace in $$...$$ blocks first
        def repl_display(m):
            inner = m.group(1)
            return '$$' + replace_in(inner) + '$$'
        text = re.sub(r'\$\$(.+?)\$\$', repl_display, text, flags=re.DOTALL)

        # Replace in $...$ inline math (avoid $$ which already handled)
        def repl_inline(m):
            inner = m.group(1)
            return '$' + replace_in(inner) + '$'
        text = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)',
                      repl_inline, text, flags=re.DOTALL)
        return text
