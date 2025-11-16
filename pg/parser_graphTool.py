"""
parserGraphTool.pl - Graph tool parser.

Top-level barrel module for short imports (1:1 parity with Perl parserGraphTool.pl).
Re-exports from pg.macros.parsers.parserGraphTool.

Reference: macros/parsers/parserGraphTool.pl
"""

from pg.macros.parsers.parserGraphTool import *

__all__ = [
    "GraphTool",
]
