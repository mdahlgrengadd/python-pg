"""GraphTool - Interactive graphing tool for WeBWorK.

This module provides the GraphTool interface used in WeBWorK problems
to create interactive graphing exercises.

Based on parserGraphTool.pl from the Perl WeBWorK distribution.
"""

from typing import Any


class GraphToolObject:
    """
    GraphTool object for interactive graphing.

    This is a stub implementation that provides the interface expected by
    WeBWorK problems using GraphTool.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize GraphTool object.

        Args:
            *args: Graph configuration arguments
            **kwargs: Graph configuration options
        """
        self.args = args
        self.kwargs = kwargs

    def with_params(self, **params):
        """
        Set parameters for the GraphTool.

        Args:
            **params: Parameters to update (bBox, grid, etc.)

        Returns:
            Self for method chaining
        """
        self.kwargs.update(params)
        return self

    def __str__(self):
        """Return string representation."""
        return "[GraphTool]"


def GraphTool(*args, **kwargs) -> GraphToolObject:
    """
    Create a GraphTool object for interactive graphing.

    GraphTool provides an interactive canvas where students can draw
    mathematical objects (points, lines, parabolas, circles, etc.) and
    have their work automatically graded.

    Args:
        *args: Configuration arguments
        **kwargs: Configuration options including:
            - bBox: Bounding box [xmin, ymax, xmax, ymin]
            - grid: Grid spacing [xgrid, ygrid]
            - axes: Axis configuration

    Returns:
        GraphToolObject instance

    Example:
        >>> gt = GraphTool()
        >>> gt = gt.with_params(bBox=[-10, 10, 10, -10], grid=[1, 1])
    """
    return GraphToolObject(*args, **kwargs)


__all__ = ['GraphTool', 'GraphToolObject']
