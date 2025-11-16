"""
parserCheckboxList.pl - Checkbox list interface.

This module provides 1:1 parity with the Perl parserCheckboxList.pl macro file.
Re-exports CheckboxList from parser_checkbox_list.

Reference: macros/parsers/parserCheckboxList.pl
"""

from .parser_checkbox_list import CheckboxList

__all__ = ["CheckboxList"]

