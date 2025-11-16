#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PG Problem Solver - Interactive command-line tool for solving PG problems.

Usage:
    python pg_solve.py <problem_file> [options]

Options:
    --seed N        Set random seed (default: random)
    --solution      Show solution after answering
    --hint          Show hint if available

Example:
    python pg_solve.py webwork_ps1_pg/ps1-prob01.pg --seed 1234 --solution
"""

import random
from pg.translator import PGTranslator
import argparse
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Ensure UTF-8 encoding for Windows terminals
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add packages to path

# Disable logging
os.environ['PYPG_DISABLE_LOGGING'] = '1'


def strip_html(html_text):
    """Remove HTML tags for terminal display."""
    import re
    # Unescape LaTeX delimiters that were escaped for HTML
    html_text = html_text.replace('\\\\(', '\\(')
    html_text = html_text.replace('\\\\)', '\\)')
    html_text = html_text.replace('\\\\[', '\\[')
    html_text = html_text.replace('\\\\]', '\\]')

    # Clean up LaTeX placeholder notation for answer blanks BEFORE removing HTML tags
    # Convert [\,\_\,] or similar patterns to just ___ for readability
    # These are visual placeholders in the problem text, not actual answer blanks
    # Match just the bracket pattern, not the surrounding \( \)
    # Use [^\].]* to ensure we don't match across sentences (periods)
    html_text = re.sub(r'\[[^\].]*\\_[^\].]*\]', '___', html_text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', html_text)
    # Decode common HTML entities
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&amp;', '&')
    text = text.replace('&nbsp;', ' ')
    # Clean up multiple spaces and newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def format_math(text):
    """Convert LaTeX math delimiters for terminal display."""
    import re

    def simplify_parens(expr):
        """Remove unnecessary parentheses around single terms in fractions."""
        # Pattern: (single_term)/(other) or (other)/(single_term)
        # Single term = number, variable, or number+greek_letter (like 23π)
        # Not a function call (no inner parens) or operation (no +/-/*)

        # Match fractions like (23π)/(6) or (x)/(2)
        def is_simple_term(s):
            """Check if string is a simple term without operators or function calls."""
            s = s.strip()
            # Has operators or function calls - keep parens
            if any(op in s for op in ['+', '-', '*', '/', ' ']):
                return False
            if '(' in s or ')' in s:
                return False
            return True

        # Remove parens from simple numerators: (simple)/(any) → simple/(any)
        expr = re.sub(
            r'\(([^()]+)\)/\(', lambda m: f'{m.group(1)}/(' if is_simple_term(m.group(1)) else m.group(0), expr)
        # Remove parens from simple denominators: (any)/(simple) → (any)/simple
        expr = re.sub(r'\)/\(([^()]+)\)', lambda m: f')/{m.group(1)}' if is_simple_term(
            m.group(1)) else m.group(0), expr)
        # Remove parens from both if both simple: (simple)/(simple) → simple/simple
        expr = re.sub(r'\(([^()]+)\)/\(([^()]+)\)',
                      lambda m: f'{m.group(1)}/{m.group(2)}' if is_simple_term(
                          m.group(1)) and is_simple_term(m.group(2)) else m.group(0),
                      expr)

        return expr

    def clean_latex(latex_str):
        """Clean up LaTeX for terminal display with mathematical notation."""
        # Remove \! (thin space) and \displaystyle
        latex_str = latex_str.replace(r'\!', '')
        latex_str = latex_str.replace(r'\displaystyle', '')

        # Convert \left( and \right) to just ( and )
        latex_str = latex_str.replace(r'\left(', '(')
        latex_str = latex_str.replace(r'\right)', ')')
        latex_str = latex_str.replace(r'\left[', '[')
        latex_str = latex_str.replace(r'\right]', ']')
        latex_str = latex_str.replace(r'\left\{', '{')
        latex_str = latex_str.replace(r'\right\}', '}')

        # Convert LaTeX bracket commands
        latex_str = latex_str.replace(r'\lbrack', '[')
        latex_str = latex_str.replace(r'\rbrack', ']')
        latex_str = latex_str.replace(r'\lbrace', '{')
        latex_str = latex_str.replace(r'\rbrace', '}')

        # Convert fractions: frac{num}{den} → (num)/(den)
        # Use regex to find nested fractions
        while 'frac{' in latex_str or r'\frac{' in latex_str:
            # Match \frac{...}{...} where ... can contain nested braces
            match = re.search(
                r'\\?frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', latex_str)
            if match:
                num = match.group(1).strip()
                den = match.group(2).strip()
                # Format as (num)/(den)
                replacement = f'({num})/({den})'
                latex_str = latex_str[:match.start()] + \
                    replacement + latex_str[match.end():]
            else:
                # Fallback: simple replacement
                latex_str = latex_str.replace(r'\frac', 'frac')
                break

        # Convert Greek letters
        latex_str = latex_str.replace(r'\pi', 'π')
        latex_str = latex_str.replace(r'\theta', 'θ')
        latex_str = latex_str.replace(r'\alpha', 'α')
        latex_str = latex_str.replace(r'\beta', 'β')
        latex_str = latex_str.replace(r'\gamma', 'γ')
        latex_str = latex_str.replace(r'\delta', 'δ')

        # Convert trig functions
        latex_str = latex_str.replace(r'\sin', 'sin')
        latex_str = latex_str.replace(r'\cos', 'cos')
        latex_str = latex_str.replace(r'\tan', 'tan')

        # Convert sqrt
        latex_str = latex_str.replace(r'\sqrt', '√')

        # Convert operators and symbols
        latex_str = latex_str.replace(r'\infty', '∞')
        latex_str = latex_str.replace(r'\cdot', '·')
        latex_str = latex_str.replace(r'\times', '×')
        latex_str = latex_str.replace(r'\div', '÷')
        latex_str = latex_str.replace(r'\pm', '±')
        latex_str = latex_str.replace(r'\leq', '≤')
        latex_str = latex_str.replace(r'\le', '≤')  # Short form
        latex_str = latex_str.replace(r'\geq', '≥')
        latex_str = latex_str.replace(r'\ge', '≥')  # Short form
        latex_str = latex_str.replace(r'\neq', '≠')
        latex_str = latex_str.replace(r'\approx', '≈')

        # Handle interval notation: [a,b[ → [a,b)  (half-open intervals)
        # Only convert the closing bracket if preceded by a comma (to avoid breaking other bracket uses)
        latex_str = re.sub(r'([0-9π]),\s*([0-9π]+)\[', r'\1, \2)', latex_str)

        # Simplify parentheses around single terms in fractions: (23π)/(6) → 23π/6
        # Helper to check if a term needs parentheses
        def needs_parens(term):
            term = term.strip()
            # Keep parens if empty or just parens
            if not term or term == '()':
                return True
            # Keep parens if term contains operators (but allow division for nested fractions)
            if any(op in term for op in ['+', '-', '*', '÷', '·', '×']):
                # Exception: if it's just a leading minus sign, that's okay
                if term.startswith('-') and not any(op in term[1:] for op in ['+', '-', '*', '÷', '·', '×']):
                    return False
                return True
            # Check for spaces, but allow spaces between numbers and Greek letters
            # (e.g., "7 π" is still a simple term)
            if ' ' in term:
                # Pattern for number followed by space and Greek letter
                if not re.match(r'^-?\d+(\.\d+)?\s*[πθαβγδ]$', term):
                    return True
            # Keep parens for function calls (has parentheses)
            if '(' in term:
                return True
            return False

        # Simplify fractions: (num)/(den) → simplified
        # Match pattern with non-greedy matching for nested parens
        def simplify_match(match):
            full = match.group(0)
            num = match.group(1).strip()
            den = match.group(2).strip()

            # Simplify
            num_display = num if needs_parens(num) else num
            den_display = den if needs_parens(den) else den

            # Rebuild
            num_final = f'({num_display})' if needs_parens(
                num) else num_display
            den_final = f'({den_display})' if needs_parens(
                den) else den_display

            return f'{num_final}/{den_final}'

        # Apply multiple times to handle nested cases
        # Use a pattern that matches balanced parentheses better
        prev = None
        while prev != latex_str:
            prev = latex_str
            # Match simple (content)/(content) where content has no unmatched parens
            latex_str = re.sub(
                r'\(([^()]+)\)/\(([^()]+)\)', simplify_match, latex_str)

        # Clean up extra spaces
        latex_str = re.sub(r'\s+', ' ', latex_str).strip()
        return latex_str

    # Inline math: \(...\) - use proper escaping
    text = re.sub(r'\\\((.+?)\\\)',
                  lambda m: f'[ {clean_latex(m.group(1))} ]', text)
    # Inline math: $...$  (but not $$)
    text = re.sub(
        r'(?<!\$)\$(?!\$)([^$]+)\$', lambda m: f'[ {clean_latex(m.group(1))} ]', text)
    # Display math: \[...\]
    text = re.sub(
        r'\\\[(.+?)\\\]', lambda m: f'\n\n  {clean_latex(m.group(1))}\n\n', text, flags=re.DOTALL)
    # Display math: $$...$$
    text = re.sub(r'\$\$(.+?)\$\$',
                  lambda m: f'\n\n  {clean_latex(m.group(1))}\n\n', text, flags=re.DOTALL)

    return text


def display_problem(result, show_solution=False, show_hint=False):
    """Display the problem statement in a readable format."""
    print("\n" + "="*70)
    print("  PROBLEM")
    print("="*70 + "\n")

    if result.statement_html:
        # Apply format_math BEFORE strip_html so LaTeX delimiters are still present
        statement = format_math(result.statement_html)
        statement = strip_html(statement)
        print(statement)
    else:
        print("(No problem statement)")

    print()

    # Show answer blanks
    if result.answer_blanks:
        print(f"This problem has {len(result.answer_blanks)} answer blank(s).")
    else:
        print("(No answer blanks detected)")

    # Show hint if requested and available
    if show_hint and result.hint_html:
        print("\n" + "-"*70)
        print("  HINT")
        print("-"*70 + "\n")
        # Apply format_math BEFORE strip_html so LaTeX delimiters are still present
        hint = format_math(result.hint_html)
        hint = strip_html(hint)
        print(hint)

    # Show solution if requested
    if show_solution and result.solution_html:
        print("\n" + "-"*70)
        print("  SOLUTION")
        print("-"*70 + "\n")
        # Apply format_math BEFORE strip_html so LaTeX delimiters are still present
        solution = format_math(result.solution_html)
        solution = strip_html(solution)
        print(solution)

    print()


def get_user_answers(result):
    """Prompt user for answers to all answer blanks."""
    answers = {}

    if not result.answer_blanks:
        return answers

    print("="*70)
    print("  ENTER YOUR ANSWERS")
    print("="*70 + "\n")

    for i, (blank_name, blank_info) in enumerate(result.answer_blanks.items(), 1):
        prompt = f"Answer {i}"
        if len(result.answer_blanks) > 1:
            prompt += f" ({blank_name})"
        prompt += ": "

        answer = input(prompt).strip()
        answers[blank_name] = answer

    return answers


def check_answers(translator, problem_file, seed, user_answers):
    """Check user's answers against the problem."""
    # Re-translate with user answers
    result = translator.translate(problem_file, seed=seed, inputs=user_answers)

    if not result.answer_results:
        print("\n❌ Unable to check answers (answer checking not available)")
        return

    print("\n" + "="*70)
    print("  RESULTS")
    print("="*70 + "\n")

    total_score = 0
    max_score = len(result.answer_results)

    for i, (blank_name, answer_result) in enumerate(result.answer_results.items(), 1):
        correct = answer_result.score >= 1.0
        user_answer = user_answers.get(blank_name, '')

        if correct:
            print(f"✓ Answer {i}: CORRECT")
            total_score += 1
        else:
            print(f"✗ Answer {i}: INCORRECT")

        print(f"  Your answer: {user_answer}")

        # Show feedback if available
        if hasattr(answer_result, 'answer_message') and answer_result.answer_message:
            print(f"  Feedback: {answer_result.answer_message}")

        print()

    # Overall score
    percentage = (total_score / max_score * 100) if max_score > 0 else 0
    print(f"Score: {total_score}/{max_score} ({percentage:.0f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive PG problem solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pg_solve.py webwork_ps1_pg/ps1-prob01.pg
  python pg_solve.py tutorial/sample-problems/Algebra/ExpandedPolynomial.pg --seed 1234
  python pg_solve.py webwork_ps1_pg/ps1-prob02.pg --solution --hint
        """
    )

    parser.add_argument('problem_file', help='Path to PG problem file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for problem generation (default: random)')
    parser.add_argument('--solution', action='store_true',
                        help='Show solution after answering')
    parser.add_argument('--hint', action='store_true',
                        help='Show hint before answering')
    parser.add_argument('--no-check', action='store_true',
                        help='Skip answer checking (just display problem)')

    args = parser.parse_args()

    # Validate problem file
    problem_path = Path(args.problem_file)
    if not problem_path.exists():
        print(f"Error: Problem file not found: {args.problem_file}")
        sys.exit(1)

    # Set seed
    seed = args.seed if args.seed is not None else random.randint(1, 99999)

    # Header
    print("\n" + "="*70)
    print(f"  PG PROBLEM SOLVER")
    print("="*70)
    print(f"\nProblem: {problem_path.name}")
    print(f"Seed: {seed}")
    if args.solution:
        print("Mode: Solution will be shown")

    try:
        # Translate problem
        translator = PGTranslator()
        result = translator.translate(str(problem_path), seed=seed)

        # Check for errors
        if result.errors:
            print(f"\n⚠️  Problem has errors:")
            for error in result.errors[:3]:
                print(f"  - {error}")
            print()

        # Display problem
        display_problem(result, show_solution=False, show_hint=args.hint)

        # If no answer blanks or --no-check, just show and exit
        if args.no_check or not result.answer_blanks:
            if args.solution:
                print("-"*70)
                print("  SOLUTION")
                print("-"*70 + "\n")

                # Show correct answers
                if result.answer_blanks:
                    print("Correct answer(s):")
                    for i, (blank_name, blank_info) in enumerate(result.answer_blanks.items(), 1):
                        label = f"Answer {i}"
                        if len(result.answer_blanks) > 1:
                            label += f" ({blank_name})"

                        # Extract correct answer from evaluator
                        # The structure is: blank_info = {"evaluator": {"ans_eval": <MathValue>}}
                        evaluator = blank_info.get("evaluator")
                        if evaluator and isinstance(evaluator, dict):
                            ans_eval = evaluator.get("ans_eval")
                            if ans_eval:
                                # Try to get the answer value
                                if hasattr(ans_eval, "TeX"):
                                    correct_ans = ans_eval.TeX()
                                elif hasattr(ans_eval, "value"):
                                    correct_ans = ans_eval.value
                                else:
                                    correct_ans = str(ans_eval)

                                # Format LaTeX math to readable ASCII (same as problem text)
                                correct_ans_str = str(correct_ans)
                                # Wrap in inline math delimiters for format_math to process
                                formatted = format_math(
                                    f"\\({correct_ans_str}\\)")
                                # Remove the [ ] brackets that format_math adds for inline math
                                formatted = formatted.strip()
                                if formatted.startswith('[') and formatted.endswith(']'):
                                    formatted = formatted[1:-1].strip()

                                print(f"  {label}: {formatted}")
                    print()

                # Show solution text if available
                if result.solution_html:
                    solution = strip_html(result.solution_html)
                    solution = format_math(solution)
                    print(solution)
                    print()
                elif not result.answer_blanks:
                    print("(No solution available)")
                    print()
            return

        # Get user answers
        user_answers = get_user_answers(result)

        # Check answers (if answer checking is implemented)
        if user_answers:
            check_answers(translator, str(problem_path), seed, user_answers)

        # Show solution if requested
        if args.solution:
            print("="*70)
            print("  SOLUTION")
            print("="*70 + "\n")

            # Show correct answers
            if result.answer_blanks:
                print("Correct answer(s):")
                for i, (blank_name, blank_info) in enumerate(result.answer_blanks.items(), 1):
                    label = f"Answer {i}"
                    if len(result.answer_blanks) > 1:
                        label += f" ({blank_name})"

                    # Extract correct answer from evaluator
                    # The structure is: blank_info = {"evaluator": {"ans_eval": <MathValue>}}
                    evaluator = blank_info.get("evaluator")
                    if evaluator and isinstance(evaluator, dict):
                        ans_eval = evaluator.get("ans_eval")
                        if ans_eval:
                            # Try to get the answer value
                            if hasattr(ans_eval, "TeX"):
                                correct_ans = ans_eval.TeX()
                            elif hasattr(ans_eval, "value"):
                                correct_ans = ans_eval.value
                            else:
                                correct_ans = str(ans_eval)

                            # Format LaTeX math to readable ASCII (same as problem text)
                            correct_ans_str = str(correct_ans)
                            # Wrap in inline math delimiters for format_math to process
                            formatted = format_math(f"\\({correct_ans_str}\\)")
                            # Remove the [ ] brackets that format_math adds for inline math
                            formatted = formatted.strip()
                            if formatted.startswith('[') and formatted.endswith(']'):
                                formatted = formatted[1:-1].strip()

                            print(f"  {label}: {formatted}")
                print()

            # Show solution text if available
            if result.solution_html:
                solution = strip_html(result.solution_html)
                solution = format_math(solution)
                print(solution)
                print()
            elif not result.answer_blanks:
                print("(No solution available)")
                print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
