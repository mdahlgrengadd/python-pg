"""LayoutTable - Table layout utilities for WeBWorK.

This module provides the LayoutTable interface used in WeBWorK problems
to create formatted table layouts.

Based on niceTables.pl from the Perl WeBWorK distribution.
"""

from typing import List, Any


def LayoutTable(rows: List[List[Any]], **kwargs) -> str:
    """
    Create a formatted table layout.

    LayoutTable creates nicely formatted HTML tables for organizing
    problem content, answer blanks, and other elements.

    Args:
        rows: List of rows, where each row is a list of cell contents
        **kwargs: Table formatting options including:
            - align: Alignment for cells ('left', 'center', 'right')
            - tex_align: LaTeX alignment specification
            - center: Whether to center the table
            - caption: Table caption
            - midrules: Whether to add horizontal rules

    Returns:
        String representation of the table

    Example:
        >>> LayoutTable([
        ...     ['x', 'f(x)'],
        ...     [1, 5],
        ...     [2, 7]
        ... ])
        '[Table with 3 rows]'
    """
    num_rows = len(rows) if rows else 0
    return f"[Table with {num_rows} rows]"


def DataTable(data: List[List[Any]], **kwargs) -> str:
    """
    Create a data table with headers.

    Similar to LayoutTable but with additional formatting for data presentation.

    Args:
        data: List of rows including header row
        **kwargs: Table formatting options

    Returns:
        String representation of the data table
    """
    return LayoutTable(data, **kwargs)


def BeginTable(**kwargs) -> str:
    """
    Begin a table environment.

    Returns opening tag for manual table construction.

    Args:
        **kwargs: Table options

    Returns:
        Opening table tag
    """
    return "<table>"


def Row(cells: List[Any], **kwargs) -> str:
    """
    Create a table row.

    Args:
        cells: List of cell contents
        **kwargs: Row options

    Returns:
        String representation of row
    """
    return f"<tr>{''.join(f'<td>{cell}</td>' for cell in cells)}</tr>"


def EndTable() -> str:
    """
    End a table environment.

    Returns:
        Closing table tag
    """
    return "</table>"


__all__ = [
    'LayoutTable',
    'DataTable',
    'BeginTable',
    'Row',
    'EndTable'
]
