"""
parserGraphTool.pl - Interactive graph tool.

This module provides 1:1 parity with the Perl parserGraphTool.pl macro file.
Re-exports GraphTool from parser_graphtool.

Reference: macros/parsers/parserGraphTool.pl
"""

from ..graph.parser_graphtool import GraphTool

__all__ = ["GraphTool"]

