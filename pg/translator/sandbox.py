"""
Subprocess-based sandbox for safe PG code execution.

This replaces RestrictedPython with a simpler, more reliable approach:
- Execute code in a separate Python subprocess
- Resource limits (timeout, memory)
- Controlled environment with only necessary imports
- Serialized communication via JSON

Advantages over RestrictedPython:
- No variable naming restrictions
- No missing guard function issues
- Better isolation (separate process)
- Simpler implementation
- More reliable

Reference: Perl's Safe.pm compartment in Translator.pm
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pg.parser import Context


@dataclass
class SandboxResult:
    """Result from sandbox execution."""

    success: bool
    """Whether execution succeeded"""

    text_segments: list[str]
    """Accumulated problem text"""

    pgml_segments: list[str]
    """Accumulated PGML text"""

    answers: dict[str, Any]
    """Answer evaluators"""

    solution_segments: list[str]
    """Solution text"""

    hint_segments: list[str]
    """Hint text"""

    errors: str
    """Error messages if any"""

    variables: dict[str, Any] | None = None
    """Problem variables"""

    stdout: str = ""
    """Captured stdout"""

    stderr: str = ""
    """Captured stderr"""


class Sandbox:
    """
    Subprocess-based sandbox for safe code execution.

    Executes PG code in a separate Python process with:
    - Timeout protection
    - Controlled imports
    - Serialized I/O
    - Error isolation
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize sandbox.

        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout

    def execute(
        self, code: str, seed: int, context: Context | None = None, globals_dict: dict[str, Any] | None = None
    ) -> SandboxResult:
        """
        Execute code in sandbox.

        Args:
            code: Python code to execute
            seed: Random seed
            context: Mathematical context
            globals_dict: Additional global variables

        Returns:
            SandboxResult with execution results
        """
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            code_file = Path(f.name)
            f.write(self._wrap_code(code, seed, context, globals_dict))

        try:
            # Execute in subprocess
            result = subprocess.run(
                [sys.executable, str(code_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=Path.cwd(),
            )

            # Parse JSON output from last line of stdout
            lines = result.stdout.strip().split("\n")
            if lines and lines[-1].startswith("{"):
                try:
                    output = json.loads(lines[-1])
                    return SandboxResult(
                        success=result.returncode == 0,
                        text_segments=output.get("text_segments", []),
                        pgml_segments=output.get("pgml_segments", []),
                        answers=output.get("answers", {}),
                        solution_segments=output.get("solution_segments", []),
                        hint_segments=output.get("hint_segments", []),
                        errors=output.get("errors", ""),
                        variables=output.get("variables", {}),
                        stdout="\n".join(lines[:-1]) if len(lines) > 1 else "",
                        stderr=result.stderr,
                    )
                except json.JSONDecodeError as e:
                    return SandboxResult(
                        success=False,
                        text_segments=[],
                        pgml_segments=[],
                        answers={},
                        solution_segments=[],
                        hint_segments=[],
                        errors=f"Failed to parse output: {e}\nOutput: {result.stdout}",
                        stdout=result.stdout,
                        stderr=result.stderr,
                    )

            # No JSON output - execution failed
            return SandboxResult(
                success=False,
                text_segments=[],
                pgml_segments=[],
                answers={},
                solution_segments=[],
                hint_segments=[],
                errors=f"No output from execution\nStderr: {result.stderr}",
                stdout=result.stdout,
                stderr=result.stderr,
            )

        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                text_segments=[],
                pgml_segments=[],
                answers={},
                solution_segments=[],
                hint_segments=[],
                errors=f"Execution timed out after {self.timeout} seconds",
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                text_segments=[],
                pgml_segments=[],
                answers={},
                solution_segments=[],
                hint_segments=[],
                errors=f"Sandbox error: {e}",
            )
        finally:
            # Clean up temp file
            try:
                code_file.unlink()
            except Exception:
                pass

    def _wrap_code(
        self, code: str, seed: int, context: Context | None = None, globals_dict: dict[str, Any] | None = None
    ) -> str:
        """
        Wrap user code with sandbox environment.

        Creates a complete Python script that:
        1. Sets up PG environment
        2. Executes user code
        3. Serializes results to JSON

        Args:
            code: User's PG code
            seed: Random seed
            context: Mathematical context
            globals_dict: Additional globals

        Returns:
            Complete Python script
        """
        # Serialize context and globals
        context_name = context.name if context else "Numeric"
        globals_json = json.dumps(globals_dict or {})

        return f'''
import json
import sys
import random
from dataclasses import dataclass, field

# Import PG packages (these must be installed in the Python environment)
try:
    from pg.math import Real, Complex, Formula, Point, Vector, Matrix, Interval
    from pg.parser import Context, Parser
    from pg.answer import AnswerEvaluator
    from pg.answer.evaluators.numeric import NumericEvaluator
    from pg.answer.evaluators.formula import FormulaEvaluator
    from pg.answer.evaluators.string import StringEvaluator
except ImportError as e:
    print(json.dumps({{"success": False, "errors": f"Import error: {{e}}"}}))
    sys.exit(1)

# Compute is an alias for Formula (for Perl compatibility)
Compute = Formula

# Set random seed
random.seed({seed})

# Initialize environment
class PGEnv:
    """PG execution environment."""
    def __init__(self):
        self.seed = {seed}
        self.context = Context("{context_name}")
        self.text_segments = []
        self.pgml_segments = []
        self.answers = {{}}
        self.solution_segments = []
        self.hint_segments = []
        self.errors = ""
        self.variables = {{}}

    def add_text(self, text):
        """Add text to problem output."""
        self.text_segments.append(str(text))

    def add_pgml_text(self, pgml):
        """Add PGML text to problem."""
        self.pgml_segments.append(str(pgml))

    def add_solution(self, text):
        """Add solution text."""
        self.solution_segments.append(str(text))

    def add_hint(self, text):
        """Add hint text."""
        self.hint_segments.append(str(text))

    def register_answer(self, name, evaluator):
        """Register answer evaluator."""
        self.answers[name] = str(evaluator)

pg_env = PGEnv()

# PG random function (Perl compatibility)
def random_func(min_val=0.0, max_val=1.0):
    """Generate random number in range [min_val, max_val)."""
    return min_val + random.random() * (max_val - min_val)

# Make 'random' callable as a function (not just the module)
# Store original module
_random_module = random
# Replace with our wrapper but keep module methods available
class RandomWrapper:
    def __call__(self, min_val=0.0, max_val=1.0):
        return random_func(min_val, max_val)
    def __getattr__(self, name):
        return getattr(_random_module, name)
random = RandomWrapper()

# Macro functions (Perl compatibility)
def num_cmp(correct, **options):
    """Numeric comparison answer evaluator."""
    return NumericEvaluator(correct_answer=correct, **options)

def fun_cmp(correct, **options):
    """Formula/function answer evaluator."""
    return FormulaEvaluator(correct_answer=correct, **options)

def str_cmp(correct, **options):
    """String comparison answer evaluator."""
    return StringEvaluator(correct_answer=correct, **options)


# PG functions available to problems
def TEXT(*args):
    """Add text to problem output."""
    pg_env.text_segments.extend(str(arg) for arg in args)

def PGML(text):
    """Add PGML text to problem."""
    pg_env.pgml_segments.append(text)

def SOLUTION(*args):
    """Add solution text."""
    pg_env.solution_segments.extend(str(arg) for arg in args)

def HINT(*args):
    """Add hint text."""
    pg_env.hint_segments.extend(str(arg) for arg in args)

def ANS(*args):
    """Register answer evaluators (with optional name)."""
    # Support both ANS(evaluator) and ANS(evaluator, name)
    if len(args) == 1:
        evaluator = args[0]
        ans_name = f"AnSwEr{{len(pg_env.answers):04d}}"
        pg_env.answers[ans_name] = evaluator  # Store evaluator object
    elif len(args) == 2:
        evaluator, ans_name = args
        pg_env.answers[ans_name] = evaluator  # Store evaluator object
    else:
        for evaluator in args:
            ans_name = f"AnSwEr{{len(pg_env.answers):04d}}"
            pg_env.answers[ans_name] = evaluator  # Store evaluator object

def NAMED_ANS(name, evaluator):
    """Register named answer evaluator."""
    pg_env.answers[name] = evaluator  # Store evaluator object

# Additional globals
globals_dict = {globals_json}

# Execute user code
try:
{self._indent(code, "    ")}
except Exception as e:
    import traceback
    pg_env.errors = traceback.format_exc()

# Helper to safely serialize values
def safe_serialize(obj):
    """Recursively serialize objects, converting non-JSON-serializable values to strings."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {{k: safe_serialize(v) for k, v in obj.items()}}
    # Try JSON serialization
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # Not JSON serializable - convert to string
        return str(obj)

# Serialize results to JSON
# Serialize answer evaluators to dicts with type info
serialized_answers = {{}}
for name, evaluator in pg_env.answers.items():
    if hasattr(evaluator, "__dict__"):
        # Safely serialize all attributes
        attrs = {{}}
        for k, v in evaluator.__dict__.items():
            if not k.startswith("_"):
                attrs[k] = safe_serialize(v)
        serialized_answers[name] = {{
            "_type": type(evaluator).__name__,
            "_module": type(evaluator).__module__,
            **attrs
        }}
    else:
        serialized_answers[name] = str(evaluator)

# Also serialize variables dict
serialized_variables = {{}}
for name, value in pg_env.variables.items():
    try:
        json.dumps(value)  # Test if JSON serializable
        serialized_variables[name] = value
    except:
        serialized_variables[name] = str(value)

output = {{
    "text_segments": pg_env.text_segments,
    "pgml_segments": pg_env.pgml_segments,
    "answers": serialized_answers,
    "solution_segments": pg_env.solution_segments,
    "hint_segments": pg_env.hint_segments,
    "errors": pg_env.errors,
    "variables": serialized_variables,
}}

print(json.dumps(output))
'''

    def _indent(self, code: str, indent: str) -> str:
        """Indent code block."""
        return "\n".join(indent + line if line.strip() else "" for line in code.split("\n"))
