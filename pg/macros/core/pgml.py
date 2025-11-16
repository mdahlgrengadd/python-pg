"""
PGML.pl - PGML rendering support

Python port of macros/core/PGML.pl
Provides PGML text rendering.

Reference: PGML.pl
"""

from pg.pgml import HTMLRenderer, PGMLParser, TeXRenderer

# Export list
__exports__ = [
    "PGML",
    "BEGIN_PGML",
    "END_PGML",
]


def PGML(text: str, **kwargs) -> str:
    """
    Render PGML text to HTML.

    Args:
        text: PGML markup
        **kwargs: Rendering options (context, etc.)

    Returns:
        Rendered HTML

    Reference: PGML.pl::PGML
    """
    # Parse PGML
    doc = PGMLParser.parse_text(text)

    # Render to HTML
    context = kwargs.get("context", {})
    renderer = HTMLRenderer(context=context)

    return renderer.render(doc)


# BEGIN_PGML and END_PGML are preprocessor directives, not functions
# They're handled by the PG preprocessor
# These are here for completeness in the macro system

def BEGIN_PGML() -> None:
    """Preprocessor directive - handled by preprocessor."""
    pass


def END_PGML() -> None:
    """Preprocessor directive - handled by preprocessor."""
    pass
