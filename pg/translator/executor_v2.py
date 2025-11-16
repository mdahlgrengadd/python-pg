"""
PG Executor V2 - Subprocess-based safe execution for PG problems.

This replaces RestrictedPython with a subprocess-based sandbox:
- Execute code in separate Python process
- No variable naming restrictions
- Better isolation and reliability
- Resource limits (timeout)

Reference: Translator.pm::set_mask() and Safe compartment
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import Any

from pg.answer import AnswerEvaluator, EvaluatorRegistry
from pg.math import Complex, Formula, Real
from pg.parser import Context, Parser
from pg.pgml import HTMLRenderer, PGMLParser

from .sandbox import Sandbox, SandboxResult


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

    errors: str = ""
    """Accumulated errors"""

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
        self.solution_segments.append(pgml)

    def add_hint(self, text: str) -> None:
        """Add plain text to hint."""
        self.hint_segments.append(text)

    def add_pgml_hint(self, pgml: str) -> None:
        """Add PGML text to hint."""
        self.hint_segments.append(pgml)

    def register_answer(self, name: str | None, evaluator: AnswerEvaluator) -> str:
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
    Execute PG code safely using subprocess sandbox.

    Provides isolated execution with:
    - Separate process isolation
    - Timeout protection
    - Controlled imports
    - No RestrictedPython limitations
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize executor.

        Args:
            timeout: Maximum execution time in seconds
        """
        self.sandbox = Sandbox(timeout=timeout)

    def execute(self, code: str, seed: int, context: Context | None = None) -> PGEnvironment:
        """
        Execute PG code in sandbox.

        Args:
            code: Python code to execute
            seed: Random seed
            context: Mathematical context (defaults to Numeric)

        Returns:
            PGEnvironment with execution results
        """
        if context is None:
            context = Context.by_name("Numeric")

        # Execute in sandbox
        result = self.sandbox.execute(code, seed, context)

        # Create environment from results
        env = PGEnvironment(seed=seed, context=context)

        if result.success:
            env.text_segments = result.text_segments
            env.pgml_segments = result.pgml_segments
            env.solution_segments = result.solution_segments
            env.hint_segments = result.hint_segments
            # Note: answers are serialized strings, not actual evaluators
            # This is a temporary limitation - will need to reconstruct evaluators
            env.errors = result.errors
        else:
            env.errors = result.errors

        return env
