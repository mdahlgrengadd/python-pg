"""
PG.pl - Core Program Generation Language functionality.

This is the Python port of macros/PG.pl (1,815 lines Perl).
Provides the fundamental macros that define the PG language.

Reference: macros/PG.pl
"""

from __future__ import annotations

import random as _random
from typing import Any, Callable


# Global PG environment (will be set by translator)
_pg_environment: "PGEnvironment | None" = None


class PGEnvironment:
    """
    PG problem environment - manages problem state.

    Equivalent to PGcore in Perl (lib/PGcore.pm)
    """

    def __init__(self, envir: dict[str, Any]):
        """
        Initialize PG environment.

        Args:
            envir: Environment dictionary with settings
        """
        self.envir = envir

        # Text accumulation arrays
        self.output_array: list[str] = []
        self.header_array: list[str] = []
        self.post_header_array: list[str] = []
        self.solution_array: list[str] = []
        self.hint_array: list[str] = []

        # Answer tracking
        self.answers_hash: dict[str, Any] = {}
        self.answer_entry_order: list[str] = []
        self.implicit_answer_queue: list[Any] = []
        self.answer_blank_queue: list[str] = []
        self.extra_answers: list[str] = []

        # Answer counter
        self._answer_counter = 0

        # Flags
        self.flags: dict[str, Any] = {
            "showPartialCorrectAnswers": envir.get("showPartialCorrectAnswers", 1),
            "recordSubmittedAnswers": envir.get("recordSubmittedAnswers", 1),
            "refreshCachedImages": envir.get("refreshCachedImages", 0),
            "solutionExists": 0,
            "hintExists": 0,
            "comment": "",
            "PROBLEM_GRADER_TO_USE": None,
        }

        # Problem settings
        self.problem_seed = envir.get("problemSeed", 123)
        self.display_mode = envir.get("displayMode", "HTML")

        # Random number generator
        self.rng = _random.Random(self.problem_seed)

        # Answer prefix
        self.answer_prefix = envir.get("ANSWER_PREFIX", "AnSwEr")
        self.quiz_prefix = envir.get("QUIZ_PREFIX", "QuIz")

        # Rendering control
        self._stop_rendering = False

    def append_text(self, text: str) -> None:
        """Append text to problem output."""
        if not self._stop_rendering:
            self.output_array.append(text)

    def append_header(self, text: str) -> None:
        """Append text to header."""
        self.header_array.append(text)

    def append_post_header(self, text: str) -> None:
        """Append text to post-header."""
        self.post_header_array.append(text)

    def new_ans_name(self) -> str:
        """
        Generate new answer name.

        Reference: PGcore.pm::new_ans_name
        """
        self._answer_counter += 1
        return f"{self.answer_prefix}{self._answer_counter:04d}"

    def record_ans_name(self, name: str, value: str = "") -> str:
        """
        Record answer name.

        Reference: PG.pl::RECORD_ANS_NAME
        """
        if name not in self.answers_hash:
            self.answers_hash[name] = {"ans_label": name}
        return name

    def record_implicit_ans_name(self, name: str) -> str:
        """
        Record implicit answer name.

        Reference: PG.pl::RECORD_IMPLICIT_ANS_NAME
        """
        self.answer_blank_queue.append(name)
        return name

    def register_answer(self, name: str, evaluator: Any) -> None:
        """
        Register answer evaluator.

        Args:
            name: Answer name/label
            evaluator: Answer evaluator object
        """
        if name not in self.answers_hash:
            self.answers_hash[name] = {}

        self.answers_hash[name]["ans_eval"] = evaluator

        if name not in self.answer_entry_order:
            self.answer_entry_order.append(name)

    def get_text(self) -> str:
        """Get accumulated problem text."""
        return "".join(self.output_array)

    def get_header(self) -> str:
        """Get accumulated header text."""
        return "".join(self.header_array)

    def get_post_header(self) -> str:
        """Get accumulated post-header text."""
        return "".join(self.post_header_array)

    def stop_rendering(self) -> None:
        """Stop accumulating output."""
        self._stop_rendering = True

    def is_rendering_stopped(self) -> bool:
        """Check if rendering is stopped."""
        return self._stop_rendering


def get_environment() -> PGEnvironment:
    """Get current PG environment."""
    global _pg_environment
    if _pg_environment is None:
        raise RuntimeError(
            "PG environment not initialized. Call DOCUMENT() first.")
    return _pg_environment


def set_environment(env: PGEnvironment) -> None:
    """Set current PG environment."""
    global _pg_environment
    _pg_environment = env


# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

def _PG_init() -> None:
    """
    Initialize PG macros in problem namespace.

    Reference: PG.pl::_PG_init (line 73)
    """
    # This is called when PG.pl is loaded
    # In Perl, this sets up MathObject context
    # For Python, we'll handle this differently
    pass


# ============================================================================
# DOCUMENT LIFECYCLE
# ============================================================================

def DOCUMENT() -> None:
    """
    Initialize problem document.

    This should be the first statement in every PG problem.

    Reference: PG.pl::DOCUMENT (line 112)
    """
    # Get environment from global or create default
    import sys
    frame = sys._getframe(1)

    # Try to get envir from caller's namespace
    envir = frame.f_globals.get("envir", {})

    # Create environment
    env = PGEnvironment(envir)
    set_environment(env)

    # Set global variables in caller's namespace
    frame.f_globals["ANSWER_PREFIX"] = env.answer_prefix
    frame.f_globals["QUIZ_PREFIX"] = env.quiz_prefix
    frame.f_globals["showPartialCorrectAnswers"] = env.flags["showPartialCorrectAnswers"]
    frame.f_globals["displayMode"] = env.display_mode
    frame.f_globals["problemSeed"] = env.problem_seed


def ENDDOCUMENT() -> tuple[str, str, str, dict[str, Any], dict[str, Any]]:
    """
    Finalize problem document.

    This must be the last statement in every PG problem.
    Returns tuple of (text, header, post_header, answers, flags).

    Reference: PG.pl::ENDDOCUMENT (line 951)
    """
    env = get_environment()

    # Finalize flags
    import sys
    frame = sys._getframe(1)

    env.flags["showPartialCorrectAnswers"] = frame.f_globals.get(
        "showPartialCorrectAnswers",
        env.flags["showPartialCorrectAnswers"]
    )

    # Get accumulated content
    text = env.get_text()
    header = env.get_header()
    post_header = env.get_post_header()

    # Set answer entry order
    env.flags["ANSWER_ENTRY_ORDER"] = env.answer_entry_order
    env.flags["KEPT_EXTRA_ANSWERS"] = env.extra_answers

    return (text, header, post_header, env.answers_hash, env.flags)


# ============================================================================
# TEXT OUTPUT FUNCTIONS
# ============================================================================

def TEXT(*args: Any) -> None:
    """
    Append text to problem output.

    Usage:
        TEXT("Problem text", "More text")

    Reference: PG.pl::TEXT (line 181)
    """
    env = get_environment()

    # Join arguments with spaces between them
    text = " ".join(str(arg) for arg in args)
    env.append_text(text)


def BEGIN_TEXT() -> str:
    """
    Begin text block marker.

    Reference: PG.pl (used with END_TEXT in preprocessor)
    """
    return ""


def END_TEXT() -> str:
    """
    End text block marker.

    Reference: PG.pl (used with BEGIN_TEXT in preprocessor)
    """
    return ""


def HEADER_TEXT(*args: Any) -> None:
    """
    Append text to HTML header.

    Usage:
        HEADER_TEXT("<script>...</script>")

    Reference: PG.pl::HEADER_TEXT (line 201)
    """
    env = get_environment()
    text = " ".join(str(arg) for arg in args)
    env.append_header(text)


def POST_HEADER_TEXT(*args: Any) -> None:
    """
    Append text after header (DEPRECATED).

    Reference: PG.pl::POST_HEADER_TEXT (line 215)
    """
    env = get_environment()
    text = " ".join(str(arg) for arg in args)
    env.append_post_header(text)


def STOP_RENDERING() -> None:
    """
    Stop accumulating output text.

    Reference: PG.pl (part of rendering control)
    """
    env = get_environment()
    env.stop_rendering()


# ============================================================================
# ANSWER FUNCTIONS
# ============================================================================

def ANS(*evaluators: Any, **kwargs) -> None:
    """
    Register answer evaluators (implicit pairing).

    Evaluators are paired with answer blanks in order.

    Usage:
        TEXT(ans_rule(), ans_rule())
        ANS(num_cmp(42), num_cmp(17))
        ANS(cmp(...), vars => ['x', 'y'])  # With options

    Args:
        *evaluators: Answer evaluators
        **kwargs: Optional keyword arguments (vars, etc.) - passed through

    Reference: PG.pl::ANS (line 469)
    """
    env = get_environment()

    for evaluator in evaluators:
        # Get next answer name from queue or create new one
        if env.answer_blank_queue:
            name = env.answer_blank_queue.pop(0)
        else:
            name = env.new_ans_name()
            # Don't record implicit name here - it's already recorded when blank is created
            # or will be recorded when we register

        env.register_answer(name, evaluator)


def NAMED_ANS(name: str, evaluator: Any) -> None:
    """
    Register named answer evaluator (explicit pairing).

    Usage:
        TEXT(NAMED_ANS_RULE("answer1"))
        NAMED_ANS("answer1", num_cmp(42))

    Reference: PG.pl::NAMED_ANS (line 432)
    """
    env = get_environment()
    env.register_answer(name, evaluator)


def LABELED_ANS(name: str, evaluator: Any) -> None:
    """
    Alias for NAMED_ANS.

    Reference: PG.pl::LABELED_ANS (line 443)
    """
    NAMED_ANS(name, evaluator)


def RECORD_ANS_NAME(name: str, value: str = "") -> str:
    """
    Record answer name.

    Used internally by answer blank macros.

    Reference: PG.pl::RECORD_ANS_NAME (line 483)
    """
    env = get_environment()
    return env.record_ans_name(name, value)


def RECORD_IMPLICIT_ANS_NAME(name: str) -> str:
    """
    Record implicit answer name.

    Reference: PG.pl::RECORD_IMPLICIT_ANS_NAME (line 499)
    """
    env = get_environment()
    return env.record_implicit_ans_name(name)


def NEW_ANS_NAME() -> str:
    """
    Generate new anonymous answer name.

    Reference: PG.pl::NEW_ANS_NAME (line 515)
    """
    env = get_environment()
    return env.new_ans_name()


def ANS_NUM_TO_NAME(num: int) -> str:
    """
    Generate answer name from number (DEPRECATED).

    Reference: PG.pl::ANS_NUM_TO_NAME (line 529)
    """
    env = get_environment()
    return f"{env.answer_prefix}{num:04d}"


def RECORD_FORM_LABEL(name: str) -> None:
    """
    Record form field name.

    Reference: PG.pl::RECORD_FORM_LABEL (line 594)
    """
    RECORD_EXTRA_ANSWERS(name)


def RECORD_EXTRA_ANSWERS(name: str) -> None:
    """
    Record extra answer field.

    Reference: PG.pl::RECORD_EXTRA_ANSWERS (line 598)
    """
    env = get_environment()
    if name not in env.extra_answers:
        env.extra_answers.append(name)


def ans_rule_count() -> int:
    """
    Get count of answer rules.

    Reference: PG.pl::ans_rule_count (line 504)
    """
    env = get_environment()
    return len(env.answers_hash)


# ============================================================================
# SOLUTION AND HINT FUNCTIONS
# ============================================================================

def SOLUTION(*args: Any) -> None:
    """
    Add solution text.

    Usage:
        SOLUTION("The answer is 42 because...")

    Reference: PG.pl (SOLUTION handling)
    """
    env = get_environment()
    env.flags["solutionExists"] = 1

    # Collect solution text
    solution_text = "".join(str(arg) for arg in args)
    env.solution_array.append(solution_text)


def HINT(*args: Any) -> None:
    """
    Add hint text.

    Usage:
        HINT("Try using the quadratic formula")

    Reference: PG.pl (HINT handling)
    """
    env = get_environment()
    env.flags["hintExists"] = 1

    # Collect hint text
    hint_text = "".join(str(arg) for arg in args)
    env.hint_array.append(hint_text)


def COMMENT(*args: Any) -> None:
    """
    Add comment (not shown to students).

    Reference: PG.pl (COMMENT handling)
    """
    env = get_environment()
    comment = " ".join(str(arg) for arg in args)
    env.flags["comment"] = comment


# ============================================================================
# MACRO LOADING
# ============================================================================

def loadMacros(*macro_files: str) -> None:
    """
    Load macro files.

    Usage:
        loadMacros("PGstandard.pl", "MathObjects.pl", "PGML.pl")

    Reference: PG.pl::loadMacros (line 1567)
    """
    # Get the macro loader from environment
    import sys
    frame = sys._getframe(1)

    # Try to get macro loader from globals
    macro_loader = frame.f_globals.get("_macro_loader")

    if macro_loader is None:
        # If no loader available, warn but don't fail
        # This allows basic testing without full translator setup
        import warnings
        warnings.warn(f"No macro loader available, cannot load: {macro_files}")
        return

    # Load each macro
    for macro_file in macro_files:
        macro_loader.load_macro(macro_file)


# ============================================================================
# PROBLEM GRADER FUNCTIONS
# ============================================================================

def install_problem_grader(grader: Callable) -> Callable:
    """
    Install custom problem grader.

    Reference: PG.pl::install_problem_grader (line 1583)
    """
    env = get_environment()
    env.flags["PROBLEM_GRADER_TO_USE"] = grader
    return grader


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def not_null(value: Any) -> bool:
    """
    Check if value is not null/empty.

    Reference: PG.pl::not_null (line 86)
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return bool(value)


def DEBUG_MESSAGE(*messages: Any) -> None:
    """
    Add debug message.

    Reference: PG.pl::DEBUG_MESSAGE (line 95)
    """
    # In full implementation, this would log debug messages
    # For now, just print to stderr
    import sys
    import traceback

    caller = traceback.extract_stack()[-2]
    print(
        f"---- {caller.filename}:{caller.lineno} in {caller.name} ------", file=sys.stderr)
    for msg in messages:
        print(msg, file=sys.stderr)
    print("__________________________", file=sys.stderr)


def WARN_MESSAGE(*messages: Any) -> None:
    """
    Add warning message.

    Reference: PG.pl::WARN_MESSAGE (line 100)
    """
    import warnings
    for msg in messages:
        warnings.warn(str(msg))


def MODES(*, HTML: str = None, TeX: str = None, PTX: str = None, **kwargs) -> str:
    """
    Return content based on current display mode.

    Usage:
        MODES(HTML="<b>bold</b>", TeX="\\textbf{bold}")

    Reference: PGbasicmacros.pl::MODES
    """
    try:
        env = get_environment()
        mode = env.display_mode
    except RuntimeError:
        mode = "HTML"

    # Map mode to argument (case-insensitive)
    mode_upper = mode.upper()

    if mode_upper == "HTML" and HTML is not None:
        return HTML
    elif mode_upper == "TEX" and TeX is not None:
        return TeX
    elif mode_upper == "PTX" and PTX is not None:
        return PTX

    # Return first available mode
    if HTML is not None:
        return HTML
    elif TeX is not None:
        return TeX
    elif PTX is not None:
        return PTX

    return ""


# ============================================================================
# RANDOM NUMBER FUNCTIONS
# ============================================================================

def random(low: float = 0, high: float = 1, step: float | None = None) -> float:
    """
    Generate random number.

    When low and high are both integers and step is None, returns an integer.
    Otherwise returns a float.

    Reference: PGauxiliaryFunctions.pl::random
    """
    env = get_environment()
    if step is not None:
        # Discrete random with explicit step
        n_steps = int((high - low) / step) + 1
        return low + env.rng.randrange(n_steps) * step
    elif isinstance(low, int) and isinstance(high, int):
        # Integer range - return random integer (Perl behavior)
        return env.rng.randint(low, high)
    else:
        # Continuous random
        return env.rng.uniform(low, high)


def non_zero_random(low: float, high: float, step: float | None = None) -> float:
    """
    Generate non-zero random number.

    Reference: PGauxiliaryFunctions.pl::non_zero_random
    """
    result = 0
    while result == 0:
        result = random(low, high, step)
    return result


def list_random(*items: Any) -> Any:
    """
    Select random item from list.

    Reference: PGauxiliaryFunctions.pl::list_random
    """
    env = get_environment()
    return env.rng.choice(items)


def random_coprime(*arrays) -> tuple:
    """
    Select random tuple of coprime numbers from arrays.

    Usage:
        (d, n) = random_coprime([2, 3, 4, 6], [1, 2, 3, ..., 12])

    Returns a tuple of numbers, one from each array, where gcd of all numbers is 1.

    Reference: PGauxiliaryFunctions.pl::random_coprime
    """
    # Import the comprehensive implementation from pg_auxiliary_functions
    from .pg_auxiliary_functions import random_coprime as _random_coprime
    return _random_coprime(*arrays)


# ============================================================================
# PERSISTENT DATA
# ============================================================================

def persistent_data(label: str, value: Any = None) -> Any:
    """
    Store/retrieve persistent data.

    Reference: PG.pl::persistent_data (line 551)
    """
    # In full implementation, this would interact with problem state storage
    # For now, just use environment
    env = get_environment()

    if not hasattr(env, "_persistent_data"):
        env._persistent_data = {}  # type: ignore

    if value is None:
        # Retrieve
        return env._persistent_data.get(label)  # type: ignore
    else:
        # Store
        env._persistent_data[label] = value  # type: ignore
        return value


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Environment
    "PGEnvironment",
    "get_environment",
    "set_environment",

    # Document lifecycle
    "DOCUMENT",
    "ENDDOCUMENT",
    "_PG_init",

    # Text output
    "TEXT",
    "BEGIN_TEXT",
    "END_TEXT",
    "HEADER_TEXT",
    "POST_HEADER_TEXT",
    "STOP_RENDERING",

    # Answers
    "ANS",
    "NAMED_ANS",
    "LABELED_ANS",
    "RECORD_ANS_NAME",
    "RECORD_IMPLICIT_ANS_NAME",
    "NEW_ANS_NAME",
    "ANS_NUM_TO_NAME",
    "RECORD_FORM_LABEL",
    "RECORD_EXTRA_ANSWERS",
    "ans_rule_count",

    # Solution/Hint
    "SOLUTION",
    "HINT",
    "COMMENT",

    # Macro loading
    "loadMacros",

    # Grading
    "install_problem_grader",

    # Utilities
    "not_null",
    "DEBUG_MESSAGE",
    "WARN_MESSAGE",

    # Random
    "random",
    "non_zero_random",
    "list_random",
    "random_coprime",

    # Persistent data
    "persistent_data",
]
