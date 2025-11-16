"""
PGbasicmacros.py - Basic PG macros for answer blanks and formatting.

Python port of macros/core/PGbasicmacros.pl
Provides answer blank functions, display constants, and basic formatting.

Reference: macros/core/PGbasicmacros.pl
"""

from __future__ import annotations

from typing import Any


# Import core functions we need
try:
    from .pg_core import (
        get_environment,
        NEW_ANS_NAME,
        RECORD_IMPLICIT_ANS_NAME,
        RECORD_ANS_NAME,
        RECORD_FORM_LABEL,
    )
except ImportError:
    # For testing/standalone use
    from pg.core import (
        get_environment,
        NEW_ANS_NAME,
        RECORD_IMPLICIT_ANS_NAME,
        RECORD_ANS_NAME,
        RECORD_FORM_LABEL,
    )


def _PGbasicmacros_init() -> None:
    """
    Initialize PGbasicmacros in problem namespace.

    Sets up display mode constants.
    Reference: PGbasicmacros.pl::_PGbasicmacros_init (line 39)
    """
    env = get_environment()
    display_mode = env.display_mode

    # Export display constants to problem namespace
    import sys
    frame = sys._getframe(1)

    # Set constants based on display mode
    frame.f_globals["PAR"] = PAR()
    frame.f_globals["BR"] = BR()
    frame.f_globals["BRBR"] = BRBR()
    frame.f_globals["LQ"] = LQ()
    frame.f_globals["RQ"] = RQ()
    frame.f_globals["BBOLD"] = BBOLD()
    frame.f_globals["EBOLD"] = EBOLD()
    frame.f_globals["BITALIC"] = BITALIC()
    frame.f_globals["EITALIC"] = EITALIC()
    frame.f_globals["BUL"] = BUL()
    frame.f_globals["EUL"] = EUL()
    frame.f_globals["BCENTER"] = BCENTER()
    frame.f_globals["ECENTER"] = ECENTER()
    frame.f_globals["HR"] = HR()
    frame.f_globals["NBSP"] = NBSP()
    frame.f_globals["PI"] = PI()
    frame.f_globals["E"] = E()


# ============================================================================
# DISPLAY MODE CONSTANTS
# ============================================================================

def PAR() -> str:
    """Paragraph break."""
    env = get_environment()
    mode = env.display_mode
    if mode == "TeX":
        return "\n\n"
    elif mode == "PTX":
        return "<p>"
    else:  # HTML
        return "<p>"


def BR() -> str:
    """Line break."""
    env = get_environment()
    mode = env.display_mode
    if mode == "TeX":
        return "\\\\"
    elif mode == "PTX":
        return "<br/>"
    else:  # HTML
        return "<br/>"


def BRBR() -> str:
    """Double line break."""
    return BR() + BR()


def LQ() -> str:
    """Left quote."""
    env = get_environment()
    mode = env.display_mode
    if mode == "TeX":
        return "``"
    else:
        return "&#8220;"  # Left double quotation mark


def RQ() -> str:
    """Right quote."""
    env = get_environment()
    mode = env.display_mode
    if mode == "TeX":
        return "''"
    else:
        return "&#8221;"  # Right double quotation mark


def BBOLD() -> str:
    """Begin bold."""
    env = get_environment()
    mode = env.display_mode
    if mode == "TeX":
        return "\\textbf{"
    else:
        return "<strong>"


def EBOLD() -> str:
    """End bold."""
    env = get_environment()
    mode = env.display_mode
    if mode == "TeX":
        return "}"
    else:
        return "</strong>"


def BITALIC() -> str:
    """Begin italic."""
    env = get_environment()
    mode = env.display_mode
    if mode == "TeX":
        return "\\textit{"
    else:
        return "<em>"


def EITALIC() -> str:
    """End italic."""
    env = get_environment()
    mode = env.display_mode
    if mode == "TeX":
        return "}"
    else:
        return "</em>"


def BUL() -> str:
    """Begin underline."""
    env = get_environment()
    mode = env.display_mode
    if mode == "TeX":
        return "\\underline{"
    else:
        return "<u>"


def EUL() -> str:
    """End underline."""
    env = get_environment()
    mode = env.display_mode
    if mode == "TeX":
        return "}"
    else:
        return "</u>"


def BCENTER() -> str:
    """Begin center."""
    env = get_environment()
    mode = env.display_mode
    if mode == "TeX":
        return "\\begin{center}"
    else:
        return '<div class="center">'


def ECENTER() -> str:
    """End center."""
    env = get_environment()
    mode = env.display_mode
    if mode == "TeX":
        return "\\end{center}"
    else:
        return "</div>"


def HR() -> str:
    """Horizontal rule."""
    env = get_environment()
    mode = env.display_mode
    if mode == "TeX":
        return "\\hrulefill"
    else:
        return "<hr/>"


def NBSP() -> str:
    """Non-breaking space."""
    env = get_environment()
    mode = env.display_mode
    if mode == "TeX":
        return "~"
    else:
        return "&nbsp;"


def PI() -> float:
    """Pi constant."""
    import math
    return math.pi


def E() -> float:
    """Euler's number."""
    import math
    return math.e


def beginproblem() -> str:
    """
    Traditional PG problem header (often empty in modern problems).

    Reference: PGbasicmacros.pl::beginproblem
    Returns empty string by default.
    """
    return ""


# ============================================================================
# ANSWER BLANK FUNCTIONS
# ============================================================================

def ans_rule(width: int = 20) -> str:
    """
    Create answer blank (text input).

    Args:
        width: Width of input field in characters (default 20)

    Returns:
        HTML for answer input field

    Reference: PGbasicmacros.pl::ans_rule (line 673)
    """
    name = NEW_ANS_NAME()
    RECORD_IMPLICIT_ANS_NAME(name)
    return NAMED_ANS_RULE(name, width)


def NAMED_ANS_RULE(name: str, width: int = 20, answer_value: str = "") -> str:
    """
    Create named answer blank.

    Args:
        name: Answer name/id
        width: Width of input field
        answer_value: Default value (for sticky answers)

    Returns:
        HTML for answer input field

    Reference: PGbasicmacros.pl::NAMED_ANS_RULE
    """
    env = get_environment()
    mode = env.display_mode

    # Get stored answer value if available
    inputs_ref = env.envir.get("inputs_ref", {})
    if not answer_value:
        answer_value = inputs_ref.get(name, "")

    # Record the answer name
    RECORD_ANS_NAME(name, answer_value)

    if mode == "TeX":
        # In TeX mode, create a rule/underline
        return f"\\underline{{\\phantom{{{'x' * width}}}}}"
    elif mode == "PTX":
        # PreTeXt format
        return f'<var name="{name}" width="{width}"/>'
    else:
        # HTML mode
        escaped_value = answer_value.replace(
            '"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
        return (
            f'<input type="text" '
            f'name="{name}" '
            f'id="{name}" '
            f'class="codeshard" '
            f'size="{width}" '
            f'value="{escaped_value}" '
            f'aria-label="answer blank"/>'
        )


def ans_box(rows: int = 5, cols: int = 80) -> str:
    """
    Create answer box (textarea).

    Args:
        rows: Number of rows
        cols: Number of columns

    Returns:
        HTML for textarea

    Reference: PGbasicmacros.pl::ans_box (line 724)
    """
    name = NEW_ANS_NAME()
    RECORD_IMPLICIT_ANS_NAME(name)
    return NAMED_ANS_BOX(name, rows, cols)


def NAMED_ANS_BOX(name: str, rows: int = 5, cols: int = 80, answer_value: str = "") -> str:
    """
    Create named answer box (textarea).

    Args:
        name: Answer name/id
        rows: Number of rows
        cols: Number of columns
        answer_value: Default value

    Returns:
        HTML for textarea

    Reference: PGbasicmacros.pl::NAMED_ANS_BOX
    """
    env = get_environment()
    mode = env.display_mode

    # Get stored answer value
    inputs_ref = env.envir.get("inputs_ref", {})
    if not answer_value:
        answer_value = inputs_ref.get(name, "")

    # Record the answer name
    RECORD_ANS_NAME(name, answer_value)

    if mode == "TeX":
        # In TeX mode, create lines
        lines = [
            "\\underline{\\phantom{" + "x" * cols + "}}" for _ in range(rows)]
        return "\n\n".join(lines)
    elif mode == "PTX":
        # PreTeXt format
        return f'<var name="{name}" form="essay"/>'
    else:
        # HTML mode
        escaped_value = answer_value.replace('<', '&lt;').replace('>', '&gt;')
        return (
            f'<textarea '
            f'name="{name}" '
            f'id="{name}" '
            f'rows="{rows}" '
            f'cols="{cols}" '
            f'aria-label="answer box"'
            f'>{escaped_value}</textarea>'
        )


def ans_radio_buttons(*options: str) -> str | list[str]:
    """
    Create radio button group.

    Args:
        *options: List of options. Use '~' prefix to mark correct answer.

    Returns:
        HTML for radio buttons (as string or list depending on context)

    Reference: PGbasicmacros.pl::ans_radio_buttons (line 680)
    """
    name = NEW_ANS_NAME()
    RECORD_IMPLICIT_ANS_NAME(name)
    buttons = NAMED_ANS_RADIO_BUTTONS(name, *options)

    env = get_environment()
    mode = env.display_mode

    if mode == "TeX":
        # Wrap in itemize environment
        buttons[0] = "\\begin{itemize}\n" + buttons[0]
        buttons[-1] += "\n\\end{itemize}\n"
    elif mode == "PTX":
        buttons[0] = '<var form="buttons">\n' + buttons[0]
        buttons[-1] += '</var>'
    else:
        # HTML - wrap in container div
        buttons[0] = (
            f'<div class="radio-buttons-container" '
            f'data-feedback-insert-element="{name}" '
            f'data-feedback-insert-method="append_content" '
            f'data-feedback-btn-add-class="ms-3">'
            + buttons[0]
        )
        buttons[-1] += "</div>"

    # Return as joined string (Perl wantarray behavior)
    return " ".join(buttons)


def NAMED_ANS_RADIO_BUTTONS(name: str, *options: str) -> list[str]:
    """
    Create named radio button group.

    Args:
        name: Answer name
        *options: List of options

    Returns:
        List of HTML strings for each radio button

    Reference: PGbasicmacros.pl::NAMED_ANS_RADIO_BUTTONS
    """
    env = get_environment()
    mode = env.display_mode

    # Get stored answer
    inputs_ref = env.envir.get("inputs_ref", {})
    answer_value = inputs_ref.get(name, "")

    # Record answer name
    RECORD_ANS_NAME(name, answer_value)

    buttons = []

    for i, option in enumerate(options):
        value = f"option_{i}"
        checked = "checked" if answer_value == value else ""

        if mode == "TeX":
            # TeX format
            marker = "$\\bigcirc$" if not checked else "$\\bullet$"
            buttons.append(f"\\item {marker} {option}")
        elif mode == "PTX":
            # PreTeXt format
            buttons.append(
                f'<choice correct="{"yes" if checked else "no"}">{option}</choice>')
        else:
            # HTML format
            buttons.append(
                f'<label>'
                f'<input type="radio" name="{name}" value="{value}" {checked}/>'
                f'{option}'
                f'</label>'
            )

    return buttons


def pop_up_list(*options: Any) -> str:
    """
    Create dropdown select list.

    Args:
        *options: List of options or dict of value=>label pairs

    Returns:
        HTML for select dropdown

    Reference: PGbasicmacros.pl (pop_up_list wraps NAMED_POP_UP_LIST)
    """
    name = NEW_ANS_NAME()
    RECORD_IMPLICIT_ANS_NAME(name)
    return NAMED_POP_UP_LIST(name, *options)


def NAMED_POP_UP_LIST(name: str, *options: Any) -> str:
    """
    Create named dropdown select list.

    Args:
        name: Answer name
        *options: List of options or dict

    Returns:
        HTML for select dropdown

    Reference: PGbasicmacros.pl::NAMED_POP_UP_LIST (line 737)
    """
    env = get_environment()
    mode = env.display_mode

    # Parse options
    if len(options) == 1 and isinstance(options[0], (list, tuple)):
        # Array format: ["opt1", "opt2", "opt3"]
        option_dict = {opt: opt for opt in options[0]}
        ordered_keys = list(options[0])
    elif len(options) == 1 and isinstance(options[0], dict):
        # Dict format: {"value1": "label1", "value2": "label2"}
        option_dict = options[0]
        ordered_keys = list(option_dict.keys())
    else:
        # Alternating key-value format: "val1", "label1", "val2", "label2"
        option_dict = dict(zip(options[::2], options[1::2]))
        ordered_keys = list(options[::2])

    # Get stored answer
    inputs_ref = env.envir.get("inputs_ref", {})
    answer_value = inputs_ref.get(name, "")

    # Record answer name
    RECORD_ANS_NAME(name, answer_value)

    if mode == "TeX":
        # In TeX, just show the value or blank
        if answer_value and answer_value in option_dict:
            return f"\\underline{{{option_dict[answer_value]}}}"
        else:
            return "\\underline{\\phantom{answer}}"
    elif mode == "PTX":
        # PreTeXt format
        return f'<var name="{name}" form="popup"/>'
    else:
        # HTML format
        html = f'<select name="{name}" id="{name}" aria-label="answer dropdown">\n'

        # Add blank option if no answer selected
        if not answer_value:
            html += '  <option value="" selected>?</option>\n'

        for key in ordered_keys:
            label = option_dict[key]
            selected = "selected" if answer_value == str(key) else ""
            html += f'  <option value="{key}" {selected}>{label}</option>\n'

        html += '</select>'
        return html


# ============================================================================
# TEXT EVALUATION FUNCTIONS
# ============================================================================

def MODES(*, HTML: str = "", TeX: str = "", PTX: str = "") -> str:
    """
    Return text based on display mode.

    Args:
        HTML: HTML mode text
        TeX: TeX mode text
        PTX: PreTeXt mode text

    Returns:
        Text for current display mode

    Reference: PGbasicmacros.pl (MODES pattern)
    """
    env = get_environment()
    mode = env.display_mode

    if "TeX" in mode:
        return TeX if TeX else HTML
    elif mode == "PTX":
        return PTX if PTX else HTML
    else:
        return HTML


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def image(filename: str, **options: Any) -> str:
    """
    Insert image.

    Args:
        filename: Image filename
        **options: width, height, tex_size, alt, etc.

    Returns:
        HTML or TeX for image

    Reference: PGbasicmacros.pl (image functions)
    """
    env = get_environment()
    mode = env.display_mode

    width = options.get("width", "")
    height = options.get("height", "")
    tex_size = options.get("tex_size", 400)
    alt = options.get("alt", "")

    if mode == "TeX":
        # TeX mode: use includegraphics
        return f"\\includegraphics[width={tex_size}pt]{{{filename}}}"
    else:
        # HTML/PTX: img tag
        attrs = []
        if width:
            attrs.append(f'width="{width}"')
        if height:
            attrs.append(f'height="{height}"')
        if alt:
            attrs.append(f'alt="{alt}"')
        else:
            attrs.append('alt="Graph"')

        attr_str = " ".join(attrs)
        return f'<img src="{filename}" {attr_str}/>'


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Initialization
    "_PGbasicmacros_init",

    # Display constants
    "PAR", "BR", "BRBR", "LQ", "RQ",
    "BBOLD", "EBOLD",
    "BITALIC", "EITALIC",
    "BUL", "EUL",
    "BCENTER", "ECENTER",
    "HR", "NBSP",
    "PI", "E",
    "beginproblem",

    # Answer blanks
    "ans_rule",
    "NAMED_ANS_RULE",
    "ans_box",
    "NAMED_ANS_BOX",
    "ans_radio_buttons",
    "NAMED_ANS_RADIO_BUTTONS",
    "pop_up_list",
    "NAMED_POP_UP_LIST",

    # Utilities
    "MODES",
    "image",
]
