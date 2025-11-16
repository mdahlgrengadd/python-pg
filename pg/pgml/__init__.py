"""
pg_pgml - PGML (PG Markup Language) parser and renderer

PGML is a markdown-like language for authoring WeBWorK problems.

Reference: macros/core/PGML.pl in legacy Perl codebase
"""

from .parser import PGMLParser
from .pgml_macros import BEGIN_PGML, END_PGML
from .pgml_macros import PGML as _PGML_MACRO  # Don't export - use sandbox version instead
from .renderer import HTMLRenderer, TeXRenderer
from .tokenizer import PGMLTokenizer, Token, TokenType

__all__ = [
    "PGMLTokenizer",
    "Token",
    "TokenType",
    "PGMLParser",
    "HTMLRenderer",
    "TeXRenderer",
    # Note: PGML is not exported here because the sandbox provides a version
    # that properly registers answer blanks with the environment.
    # "PGML" is intentionally omitted.
    "BEGIN_PGML",
    "END_PGML",
]
