"""
⚠️  DEPRECATED MODULE - DO NOT USE FOR NEW CODE ⚠️

This module is kept only for backward compatibility with existing code.
The `PGMLRenderer` class is still used by `pg.translator` but will be
replaced in a future version.

For new code, use:
- `pg.translator.PGTranslator` for full problem translation (recommended)
- `pg.pgml.renderer.HTMLRenderer` for PGML rendering (future replacement)

This module will be removed in a future version once `pg.translator` is
refactored to use `pg.pgml.renderer.HTMLRenderer` directly.
"""

from typing import TYPE_CHECKING

# Only export PGMLRenderer - it's still used by pg.translator
from .pgml import PGMLRenderer

if TYPE_CHECKING:
    from typing import Any

__all__ = ['PGMLRenderer']
