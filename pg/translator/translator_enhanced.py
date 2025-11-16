"""
Compatibility shim for EnhancedPGTranslator.

The enhanced implementation now lives directly in pg.translator.translator.
This module keeps the older import path working without duplicating code.
"""

from __future__ import annotations

from .translator import PGTranslator, ProblemResult

EnhancedPGTranslator = PGTranslator

__all__ = ["EnhancedPGTranslator", "ProblemResult"]
