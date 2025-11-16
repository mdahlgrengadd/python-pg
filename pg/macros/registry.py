"""
Macro Registry for PG Translator.

Defines which macros are loaded by default (core) vs on-demand (optional).
This enables dynamic runtime loading similar to the Perl reference implementation.
"""

from __future__ import annotations

from typing import Any


# Core macros that are always pre-loaded for every problem
# These are essential functions that nearly all problems use
CORE_MACROS = {
    "essential": {
        "module": None,  # Built into sandbox
        "functions": [
            "DOCUMENT",
            "TEXT",
            "ANS",
            "Compute",
            "Real",
            "Context",
            "Formula",
            "String",
            "loadMacros",  # Must be available to load other macros!
        ],
        "description": "Essential PG functions required by all problems",
    },
}

# Optional macros loaded on-demand via loadMacros()
# Organized by category for easier management
OPTIONAL_MACROS = {
    # Core PG functionality
    "PG": {
        "module": "pg.pg",
        "aliases": ["PG.pl"],
        "category": "core",
        "functions": [],  # Empty = import entire module
        "description": "Core PG macro functions (1:1 parity with PG.pl)",
    },
    # PGstandard.pl - Loads PG.pl + PGbasicmacros.pl + PGanswermacros.pl
    "PGstandard": {
        "module": "pg.standard",
        "aliases": ["PGstandard.pl"],
        "category": "core",
        "functions": [],  # Empty = import entire module
        "description": "Standard PG functions (1:1 parity with PGstandard.pl)",
    },
    "PGbasicmacros": {
        "module": "pg.basicmacros",
        "aliases": ["PGbasicmacros.pl"],
        "category": "core",
        "functions": [],  # Empty = import entire module
        "description": "Basic UI and formatting macros (1:1 parity with PGbasicmacros.pl)",
    },
    "PGanswermacros": {
        "module": "pg.answermacros",
        "aliases": ["PGanswermacros.pl"],
        "category": "answers",
        "functions": [],  # Empty = import entire module
        "description": "Answer evaluation macros (1:1 parity with PGanswermacros.pl)",
    },
    "answerHints": {
        "module": "pg.macros.answers.answer_hints",
        "aliases": ["answerHints.pl"],
        "category": "answers",
        "functions": ["AnswerHints"],
        "description": "Custom answer hints for student feedback (1:1 parity with answerHints.pl)",
    },
    "PGcourse": {
        "module": "pg.course",
        "aliases": ["PGcourse.pl"],
        "category": "core",
        "functions": [],  # Empty = import entire module
        "description": "Course-specific configuration (1:1 parity with PGcourse.pl)",
    },
    "PGauxiliaryFunctions": {
        "module": "pg.macros.core.pg_auxiliary_functions",
        "aliases": ["PGauxiliaryFunctions.pl"],
        "category": "core",
        "functions": [],  # Empty = import entire module
        "description": "Auxiliary mathematical functions (1:1 parity with PGauxiliaryFunctions.pl)",
    },

    # PGML
    "PGML": {
        "module": "pg.pgml",
        "aliases": ["PGML.pl"],
        "category": "markup",
        "functions": [],  # Empty = import entire module
        "description": "PG Markup Language (1:1 parity with PGML.pl)",
    },

    # MathObjects
    "MathObjects": {
        "module": "pg.mathobjects",
        "aliases": ["MathObjects.pl"],
        "category": "core",
        "functions": [],  # Empty = import entire module
        "description": "Math Object system (1:1 parity with MathObjects.pl)",
    },

    # Parser macros
    "parserPopUp": {
        "module": "pg.parser_popUp",
        "aliases": ["parserPopUp.pl"],
        "category": "parsers",
        "functions": [],  # Empty = import entire module
        "description": "Popup menu answer type (1:1 parity with parserPopUp.pl)",
    },
    "parserRadioButtons": {
        "module": "pg.parser_radioButtons",
        "aliases": ["parserRadioButtons.pl"],
        "category": "parsers",
        "functions": [],  # Empty = import entire module
        "description": "Radio button answer type (1:1 parity with parserRadioButtons.pl)",
    },
    "parserCheckboxes": {
        "module": "pg.parser_checkboxList",
        "aliases": ["parserCheckboxes.pl"],
        "category": "parsers",
        "functions": [],  # Empty = import entire module
        "description": "Checkbox answer type (1:1 parity with parserCheckboxList.pl)",
    },
    "parserMultiAnswer": {
        "module": "pg.parser_multiAnswer",
        "aliases": ["parserMultiAnswer.pl"],
        "category": "parsers",
        "functions": [],  # Empty = import entire module
        "description": "Multiple related answers (1:1 parity with parserMultiAnswer.pl)",
    },
    "parserVectorUtils": {
        "module": "pg.macros.math.vector_utils",
        "aliases": ["parserVectorUtils.pl"],
        "category": "parsers",
        "functions": [
            "non_zero_vector",
            "non_zero_vector2D",
            "non_zero_vector3D",
            "non_zero_point",
            "non_zero_point2D",
            "non_zero_point3D",
            "Overline",
            "BoldMath",
            "GRAD",
            "Plane",
            "Line",
            "norm",
            "unit",
        ],
        "description": "Vector and plane utilities (1:1 parity with parserVectorUtils.pl)",
    },
    "parserAssignment": {
        "module": "pg.macros.parsers.parser_assignment",
        "aliases": ["parserAssignment.pl"],
        "category": "parsers",
        "functions": [
            "AssignmentValue",
            "AssignmentBOP",
            "AssignmentFunction",
            "AssignmentParser",
            "Assignment",
        ],
        "description": "Assignment expressions for variable/function definitions (1:1 parity with parserAssignment.pl)",
    },
    "parserFunction": {
        "module": "pg.macros.parsers.parser_function",
        "aliases": ["parserFunction.pl"],
        "category": "parsers",
        "functions": ["parserFunction"],
        "description": "Custom function definition for contexts (1:1 parity with parserFunction.pl)",
    },
    "parserFormulaWithUnits": {
        "module": "pg.macros.parsers.parser_formula_with_units",
        "aliases": ["parserFormulaWithUnits.pl"],
        "category": "parsers",
        "functions": ["FormulaWithUnits", "FormulaWithUnits_factory"],
        "description": "Formula with physical units (Phase 2 of contextUnits)",
    },
    "parserNumberWithUnits": {
        "module": "pg.macros.parsers.parser_formula_with_units",
        "aliases": ["parserNumberWithUnits.pl"],
        "category": "parsers",
        "functions": ["NumberWithUnits", "NumberWithUnits_factory"],
        "description": "Number with physical units (Phase 2 of contextUnits)",
    },

    # Graphics macros (heavy - should be lazy loaded!)
    "PGgraphmacros": {
        "module": "pg.graphmacros",
        "aliases": ["PGgraphmacros.pl"],
        "category": "graphics",
        "functions": [],  # Empty = import entire module
        "description": "2D graphing functions (1:1 parity with PGgraphmacros.pl)",
        "lazy": True,  # Heavy dependency
    },
    "parserGraphTool": {
        "module": "pg.parser_graphTool",
        "aliases": ["parserGraphTool.pl"],
        "category": "graphics",
        "functions": [],  # Empty = import entire module
        "description": "Interactive graph tool (1:1 parity with parserGraphTool.pl)",
        "lazy": True,
    },
    "VectorField3D": {
        "module": "pg.macros.graph.vector_field_3d",
        "aliases": ["VectorField3D.pl"],
        "category": "graphics",
        "functions": ["VectorField3D"],
        "description": "3D vector field visualization",
        "lazy": True,
    },
    "plots": {
        "module": "pg.macros.graph.plots",
        "aliases": ["plots.pl"],
        "category": "graphics",
        "functions": ["Plot", "PlotObject", "PlotData", "PlotAxes"],
        "description": "Modern plotting with parametric curves (1:1 parity with plots.pl)",
        "lazy": True,
    },

    # Math macros
    "PGstatisticsmacros": {
        "module": "pg.macros.math.statistics_utils",
        "aliases": ["PGstatisticsmacros.pl"],
        "category": "math",
        "functions": [
            "stats_mean",
            "stats_sd",
            "stats_SX_SXX",
            "linear_regression",
        ],
        "description": "Statistical functions",
    },

    # Interactive macros
    "draggableProof": {
        "module": "pg.macros.math.draggable_proof",
        "aliases": ["draggableProof.pl"],
        "category": "interactive",
        "functions": ["DraggableProof"],
        "description": "Draggable proof problems",
        "lazy": True,
    },
    "draggableSubsets": {
        "module": "pg.macros.math.draggable_subsets",
        "aliases": ["draggableSubsets.pl"],
        "category": "interactive",
        "functions": ["DraggableSubsets"],
        "description": "Draggable subset selection",
        "lazy": True,
    },

    # Context macros
    "contextFraction": {
        "module": "pg.math.fraction",
        "aliases": ["contextFraction.pl"],
        "category": "contexts",
        "functions": ["Fraction"],
        "description": "Fraction context for exact fractional answers (1:1 parity with contextFraction.pl)",
    },
    "contextUnits": {
        "module": "pg.macros.contexts.context_units",
        "aliases": ["contextUnits.pl"],
        "category": "contexts",
        "functions": ["Context_Units", "UnitsContext"],
        "description": "Units context for answers with physical units (Phase 1: length, time)",
    },
    # Note: contextLimitedPolynomial and other contexts will be added in future phases
}


def get_macro_info(macro_name: str) -> dict[str, Any] | None:
    """
    Get information about a macro.

    Args:
        macro_name: Macro name (e.g., "parserPopUp", "PG.pl")

    Returns:
        Macro information dict or None if not found
    """
    # Check optional macros
    if macro_name in OPTIONAL_MACROS:
        return OPTIONAL_MACROS[macro_name]

    # Check aliases
    for key, info in OPTIONAL_MACROS.items():
        if "aliases" in info and macro_name in info["aliases"]:
            return info

    # Check core macros
    if macro_name in CORE_MACROS:
        return CORE_MACROS[macro_name]

    return None


def is_core_macro(macro_name: str) -> bool:
    """
    Check if a macro is a core macro (pre-loaded).

    Args:
        macro_name: Macro name

    Returns:
        True if core macro, False otherwise
    """
    return macro_name in CORE_MACROS


def should_lazy_load(macro_name: str) -> bool:
    """
    Check if a macro should be lazy-loaded (not pre-loaded).

    Args:
        macro_name: Macro name

    Returns:
        True if should be lazy-loaded, False if should be pre-loaded
    """
    info = get_macro_info(macro_name)
    if info is None:
        return True  # Unknown macros are lazy-loaded

    return info.get("lazy", False)


def get_macros_by_category(category: str) -> list[str]:
    """
    Get all macro names in a category.

    Args:
        category: Category name (e.g., "graphics", "parsers")

    Returns:
        List of macro names in that category
    """
    return [
        name
        for name, info in OPTIONAL_MACROS.items()
        if info.get("category") == category
    ]


def get_all_macro_names() -> list[str]:
    """
    Get all registered macro names.

    Returns:
        List of all macro names (including aliases)
    """
    names = list(CORE_MACROS.keys()) + list(OPTIONAL_MACROS.keys())

    # Add aliases
    for info in OPTIONAL_MACROS.values():
        if "aliases" in info:
            names.extend(info["aliases"])

    return names
