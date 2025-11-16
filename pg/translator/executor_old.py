"""
PG Executor - Safe execution environment for PG problems.

Uses RestrictedPython to execute PG code safely, providing:
- Safe built-ins only
- PG-specific functions (random, Formula, Compute, etc.)
- Text accumulation
- Answer registration

Reference: Translator.pm::set_mask() and Safe compartment
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import Any, Callable

from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    safe_builtins,
    safer_getattr,
)

from pg.answer import AnswerEvaluator, EvaluatorRegistry
from pg.math import Complex, Formula, Real
from pg.parser import Context, Parser
from pg.pgml import HTMLRenderer, PGMLParser


@dataclass
class PGEnvironment:
    """
    Execution environment for PG problems.

    Collects problem text, answers, solutions, and hints during execution.
    """

    seed: int
    """Random seed for problem"""

    context: Context
    """Current mathematical context"""

    text_segments: list[str] = field(default_factory=list)
    """Accumulated problem text"""

    pgml_segments: list[str] = field(default_factory=list)
    """Accumulated PGML text"""

    solution_segments: list[str] = field(default_factory=list)
    """Accumulated solution text"""

    hint_segments: list[str] = field(default_factory=list)
    """Accumulated hint text"""

    answers: dict[str, AnswerEvaluator] = field(default_factory=dict)
    """Registered answer evaluators by name"""

    variables: dict[str, Any] = field(default_factory=dict)
    """Problem variables"""

    answer_counter: int = 1
    """Counter for auto-generated answer names"""

    def add_text(self, text: str) -> None:
        """Add plain text to problem statement."""
        self.text_segments.append(text)

    def add_pgml_text(self, pgml: str) -> None:
        """Add PGML text to problem statement."""
        self.pgml_segments.append(pgml)

    def add_solution(self, text: str) -> None:
        """Add plain text to solution."""
        self.solution_segments.append(text)

    def add_pgml_solution(self, pgml: str) -> None:
        """Add PGML text to solution."""
        # For now, treat as plain text (will render later)
        self.solution_segments.append(pgml)

    def add_hint(self, text: str) -> None:
        """Add plain text to hint."""
        self.hint_segments.append(text)

    def add_pgml_hint(self, pgml: str) -> None:
        """Add PGML text to hint."""
        # For now, treat as plain text (will render later)
        self.hint_segments.append(pgml)

    def register_answer(
        self, name: str | None, evaluator: AnswerEvaluator
    ) -> str:
        """
        Register an answer evaluator.

        Args:
            name: Answer name (auto-generated if None)
            evaluator: Answer evaluator

        Returns:
            Answer name
        """
        if name is None:
            name = f"AnSwEr{self.answer_counter:04d}"
            self.answer_counter += 1

        self.answers[name] = evaluator
        return name

    def render_text(self) -> str:
        """Render all text segments to HTML."""
        # Combine plain text and PGML
        all_text = "\n".join(self.text_segments)

        # Render PGML segments
        pgml_html = ""
        if self.pgml_segments:
            pgml_text = "\n\n".join(self.pgml_segments)
            doc = PGMLParser.parse_text(pgml_text)
            renderer = HTMLRenderer(context=self.variables)
            pgml_html = renderer.render(doc)

        # Combine
        if all_text and pgml_html:
            return all_text + "\n" + pgml_html
        return all_text + pgml_html

    def render_solution(self) -> str | None:
        """Render solution text to HTML."""
        if not self.solution_segments:
            return None
        return "\n".join(self.solution_segments)

    def render_hint(self) -> str | None:
        """Render hint text to HTML."""
        if not self.hint_segments:
            return None
        return "\n".join(self.hint_segments)


class PGExecutor:
    """
    Execute PG code safely with RestrictedPython.

    Provides a sandboxed environment with:
    - Limited built-ins (no file I/O, network, etc.)
    - PG-specific functions
    - Text and answer accumulation
    """

    def __init__(self):
        """Initialize executor with safe globals."""
        self.parser = Parser()

    def execute(self, code: str, seed: int, context: Context | None = None) -> PGEnvironment:
        """
        Execute PG code in a safe sandbox.

        Args:
            code: Preprocessed Python code
            seed: Random seed
            context: Mathematical context (defaults to Numeric)

        Returns:
            PGEnvironment with collected text, answers, etc.

        Raises:
            SyntaxError: If code has syntax errors
            RuntimeError: If code execution fails
        """
        # Create environment
        if context is None:
            from pg.parser import Context as ParserContext
            context = ParserContext.numeric()  # Default context

        env = PGEnvironment(seed=seed, context=context)

        # Set random seed
        _random.seed(seed)

        # Build safe globals
        safe_builtins = self._build_safe_globals(env)

        # Compile with RestrictedPython
        try:
            byte_code = compile_restricted(code, filename="<pg>", mode="exec")
        except SyntaxError as e:
            raise SyntaxError(f"Compilation error: {e}") from e

        # Execute
        try:
            exec(byte_code, safe_builtins)
        except Exception as e:
            raise RuntimeError(f"Execution error: {e}") from e

        return env

    def _build_safe_globals(self, env: PGEnvironment) -> dict[str, Any]:
        """
        Build safe global namespace for execution.

        Includes:
        - Safe built-ins only
        - PG functions
        - Math utilities
        """
        # Start with RestrictedPython safe globals
        globals_dict = safe_globals.copy()

        # Add safe built-ins from RestrictedPython
        globals_dict.update(
            {
                "__builtins__": safe_builtins,
                # RestrictedPython guards
                "_getattr_": safer_getattr,
                "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
                "_write_": lambda x: x,  # Allow all writes (we control the sandbox)
                "_print_": lambda x: x,  # Allow print
                "__name__": "pg_problem",
                "__metaclass__": type,
                # Environment (use pg_env instead of _env for RestrictedPython)
                "pg_env": env,
            }
        )

        # Add PG-specific functions
        globals_dict.update(self._build_pg_functions(env))

        return globals_dict

    def _build_pg_functions(self, env: PGEnvironment) -> dict[str, Callable]:
        """Build PG-specific functions."""

        def random_wrapper(a: float = 0.0, b: float = 1.0, step: float | None = None) -> float:
            """
            Generate random number.

            Args:
                a: Lower bound (inclusive)
                b: Upper bound (exclusive)
                step: Step size (if provided, returns a + step * random_int)

            Returns:
                Random float in [a, b)
            """
            if step is not None:
                # Random integer step
                n = int((b - a) / step)
                return a + step * _random.randint(0, n)
            return _random.uniform(a, b)

        def formula_wrapper(expr: str, variables: list[str] | None = None) -> Formula:
            """
            Create a Formula from expression string.

            Args:
                expr: Mathematical expression
                variables: Variable names (auto-detected if None)

            Returns:
                Formula object
            """
            return Formula(expr, variables=variables, context=env.context)

        def compute_wrapper(expr: str):
            """
            Compute a mathematical expression.

            Alias for Formula (Perl compatibility).
            """
            return formula_wrapper(expr)

        def real_wrapper(value: float | int | str) -> Real:
            """Create a Real value."""
            if isinstance(value, str):
                # Parse as number
                value = float(value)
            return Real(float(value))

        def complex_wrapper(real: float, imag: float = 0) -> Complex:
            """Create a Complex value."""
            return Complex(real, imag)

        def text_wrapper(text: str) -> None:
            """
            Add text to problem (Perl TEXT function).

            Args:
                text: Text to add
            """
            env.add_text(text)

        def pgml_wrapper(pgml: str) -> None:
            """
            Add PGML text to problem.

            Args:
                pgml: PGML markup
            """
            env.add_pgml_text(pgml)

        def ans_wrapper(evaluator: AnswerEvaluator, name: str | None = None) -> None:
            """
            Register an answer evaluator (Perl ANS function).

            Args:
                evaluator: Answer evaluator
                name: Answer name (auto-generated if None)
            """
            env.register_answer(name, evaluator)

        def named_ans_wrapper(name: str, evaluator: AnswerEvaluator) -> None:
            """
            Register a named answer evaluator.

            Args:
                name: Answer name
                evaluator: Answer evaluator
            """
            env.register_answer(name, evaluator)

        # Expose answer evaluator creation functions
        def num_cmp(
            correct: float | str,
            tol_type: str = "relative",
            tolerance: float = 0.001,
            **options,
        ) -> AnswerEvaluator:
            """Create numeric answer evaluator (Perl num_cmp)."""
            from pg.answer.evaluators.numeric import NumericEvaluator

            return NumericEvaluator(
                correct_answer=correct,
                tolerance=tolerance,
                tolerance_mode=tol_type,
                **options,
            )

        def fun_cmp(
            correct: str | Formula,
            variables: list[str] | None = None,
            **options,
        ) -> AnswerEvaluator:
            """Create formula answer evaluator (Perl fun_cmp)."""
            from pg.answer.evaluators.formula import FormulaEvaluator

            if isinstance(correct, str):
                correct = formula_wrapper(correct, variables)

            return FormulaEvaluator(correct_answer=correct, **options)

        def str_cmp(
            correct: str,
            case_sensitive: bool = True,
            **options,
        ) -> AnswerEvaluator:
            """Create string answer evaluator (Perl str_cmp)."""
            from pg.answer.evaluators.string import StringEvaluator

            return StringEvaluator(
                correct_answer=correct,
                case_sensitive=case_sensitive,
                **options,
            )

        return {
            # Random number generation
            "random": random_wrapper,
            # MathObject creation
            "Formula": formula_wrapper,
            "Compute": compute_wrapper,
            "Real": real_wrapper,
            "Complex": complex_wrapper,
            # Text accumulation
            "TEXT": text_wrapper,
            "PGML": pgml_wrapper,
            # Answer registration
            "ANS": ans_wrapper,
            "NAMED_ANS": named_ans_wrapper,
            # Answer evaluators
            "num_cmp": num_cmp,
            "fun_cmp": fun_cmp,
            "str_cmp": str_cmp,
            # Math modules (for import)
            "Real": Real,
            "Complex": Complex,
            "Formula": Formula,
        }
