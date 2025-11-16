"""
pg.macros - Macro system for WeBWorK PG

Provides macro functions and utilities for problem authoring.
The macro loading system uses preprocessor-based imports via the registry.
"""

from .registry import (
    CORE_MACROS,
    OPTIONAL_MACROS,
    get_macro_info,
    is_core_macro,
    should_lazy_load,
    get_macros_by_category,
    get_all_macro_names,
)

__all__ = [
    "CORE_MACROS",
    "OPTIONAL_MACROS",
    "get_macro_info",
    "is_core_macro",
    "should_lazy_load",
    "get_macros_by_category",
    "get_all_macro_names",
]
