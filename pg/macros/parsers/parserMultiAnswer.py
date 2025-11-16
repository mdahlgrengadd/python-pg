"""
parserMultiAnswer.pl - Multiple related answers.

This module provides 1:1 parity with the Perl parserMultiAnswer.pl macro file.
Re-exports MultiAnswer from parser_multianswer.

Reference: macros/parsers/parserMultiAnswer.pl
"""

from .parser_multianswer import MultiAnswer

__all__ = ["MultiAnswer"]

