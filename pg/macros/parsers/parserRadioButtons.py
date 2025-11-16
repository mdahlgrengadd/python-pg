"""
parserRadioButtons.pl - Radio button interface.

This module provides 1:1 parity with the Perl parserRadioButtons.pl macro file.
Re-exports RadioButtons from parser_popup.

Reference: macros/parsers/parserRadioButtons.pl
"""

from .parser_popup import RadioButtons

__all__ = ["RadioButtons"]

