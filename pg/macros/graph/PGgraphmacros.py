"""
PGgraphmacros.pl - 2D graphing functions.

This module provides 1:1 parity with the Perl PGgraphmacros.pl macro file.
Re-exports graphing functions from pg.graph.

Reference: macros/graph/PGgraphmacros.pl
"""

from .pg_graph import init_graph, add_functions, Plot, WWPlot, Label, Fun

__all__ = ["init_graph", "add_functions", "Plot", "WWPlot", "Label", "Fun"]
