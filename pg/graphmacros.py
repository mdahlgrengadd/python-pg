"""
PGgraphmacros.pl - Graphing functions.

Top-level barrel module for short imports (1:1 parity with Perl PGgraphmacros.pl).
Re-exports from pg.macros.graph.PGgraphmacros.

Usage:
    from pg.graphmacros import init_graph, add_functions, Plot
    graph = init_graph(-5, -5, 5, 5)

Reference: macros/graph/PGgraphmacros.pl
"""

from pg.macros.graph.PGgraphmacros import *

__all__ = [
    "init_graph",
    "add_functions",
    "Plot",
    "WWPlot",
    "Label",
    "Fun",
]
