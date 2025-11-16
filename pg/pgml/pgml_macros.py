"""
PGML Macros - High-level interface for PGML processing.

Provides PGML() function that mimics the Perl PGML.pl interface.

Reference: macros/core/PGML.pl
"""

from __future__ import annotations

from typing import Any

from .parser import PGMLParser
from .renderer import HTMLRenderer, TeXRenderer


def PGML(text: str, **options: Any) -> str:
    """
    Parse and render PGML text to HTML.

    This is the main interface function that mimics Perl's PGML() macro.

    Args:
        text: PGML markup text to parse
        **options: Additional options (context variables, display mode, etc.)

    Returns:
        Rendered HTML string

    Usage:
        html = PGML(r'''
        Compute [`[$a] \times [$b]`].

        Answer: [_____]{$ans}
        ''')

    Reference: PGML.pl::PGML (main entry point)
    """
    # Extract context variables from options
    context = options.get("context", {})
    display_mode = options.get("display_mode", "HTML")

    # Parse PGML text
    doc = PGMLParser.parse_text(text)

    # Render based on display mode
    if display_mode.upper() == "TEX":
        renderer = TeXRenderer(context=context)
    else:
        renderer = HTMLRenderer(context=context)

    return renderer.render(doc)


def BEGIN_PGML() -> str:
    """
    Begin PGML block marker.

    In Perl, BEGIN_PGML/END_PGML are used with preprocessor.
    In Python, we use PGML() function instead.

    Reference: PGML.pl::BEGIN_PGML
    """
    return ""


def END_PGML() -> str:
    """
    End PGML block marker.

    In Perl, BEGIN_PGML/END_PGML are used with preprocessor.
    In Python, we use PGML() function instead.

    Reference: PGML.pl::END_PGML
    """
    return ""
