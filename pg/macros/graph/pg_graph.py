"""PGgraphmacros - Graph initialization and function plotting for WeBWorK.

This module provides core graphing functionality from PGgraphmacros.pl:
- init_graph(): Create and configure a graph canvas with axes and grid
- add_functions(): Plot mathematical functions on a graph
- Plot(): Wrapper for plotting functions

Based on macros/graph/PGgraphmacros.pl from the WeBWorK distribution.
"""

from typing import Any, Dict, List, Optional, Tuple, Union


class WWPlot:
    """
    Graph canvas object for plotting in WeBWorK.

    This is a Python wrapper around the Perl WWPlot class that manages
    graph initialization, axes, grids, and function plotting.

    Attributes:
        xmin, xmax, ymin, ymax: Graph bounds in real coordinates
        width, height: Canvas size in pixels
    """

    def __init__(self, width: int = 200, height: int = 200):
        """
        Initialize a graph canvas.

        Args:
            width: Canvas width in pixels (default 200)
            height: Canvas height in pixels (default 200)
        """
        self.width = width
        self.height = height
        self.xmin = -10.0
        self.xmax = 10.0
        self.ymin = -10.0
        self.ymax = 10.0
        self.ext = '.png'
        self.image_name: Optional[str] = None
        self.functions: List[Any] = []
        self.labels: List['Label'] = []
        self.stamps: List[Any] = []
        self.axes_config: Dict[str, Any] = {}
        self.grid_config: Dict[str, Any] = {}
        self.tick_config: Dict[str, Any] = {}
        self.plot_vertical_axis = True

    def imageName(self, name: str) -> None:
        """Set the image filename for this graph."""
        self.image_name = name

    def h_axis(self, level: float, color: str = 'black') -> None:
        """
        Draw horizontal (x) axis.

        Args:
            level: y-coordinate where axis appears (usually 0)
            color: Color of axis line
        """
        self.axes_config['h_axis'] = {'level': level, 'color': color}

    def v_axis(self, level: float, color: str = 'black') -> None:
        """
        Draw vertical (y) axis.

        Args:
            level: x-coordinate where axis appears (usually 0)
            color: Color of axis line
        """
        self.axes_config['v_axis'] = {'level': level, 'color': color}

    def h_grid(self, color: str, *x_values: float) -> None:
        """
        Draw horizontal grid lines.

        Args:
            color: Color of grid lines
            x_values: x-coordinates where vertical grid lines appear
        """
        self.grid_config['h_grid'] = {'color': color, 'values': x_values}

    def v_grid(self, color: str, *y_values: float) -> None:
        """
        Draw vertical grid lines.

        Args:
            color: Color of grid lines
            y_values: y-coordinates where horizontal grid lines appear
        """
        self.grid_config['v_grid'] = {'color': color, 'values': y_values}

    def h_ticks(self, level: float, color: str, *y_values: float) -> None:
        """
        Draw horizontal tick marks.

        Args:
            level: y-coordinate where ticks appear
            color: Color of tick marks
            y_values: y-coordinates of tick positions
        """
        self.tick_config['h_ticks'] = {
            'level': level, 'color': color, 'values': y_values}

    def v_ticks(self, level: float, color: str, *x_values: float) -> None:
        """
        Draw vertical tick marks.

        Args:
            level: x-coordinate where ticks appear
            color: Color of tick marks
            x_values: x-coordinates of tick positions
        """
        self.tick_config['v_ticks'] = {
            'level': level, 'color': color, 'values': x_values}

    def lb(self, label: 'Label') -> None:
        """
        Add a label to the graph.

        Args:
            label: Label object to add
        """
        self.labels.append(label)

    def stamps(self, *stamp_objects: Any) -> None:
        """
        Add visual marks/stamps to the graph (circles, points, etc.).

        Args:
            stamp_objects: Stamp objects to add
        """
        self.stamps.extend(stamp_objects)

    def add_function(self, *args, **kwargs) -> None:
        """
        Add a function to the graph from a string specification.

        This is the Perl-compatible method for adding functions. The string
        can contain function definitions with domain and style specifications.

        Args:
            *args: Function specification (string or callable)
                Example: "x^2 for x in [-5,5] using color:blue"
            **kwargs: Additional options (color, weight, etc.)

        Example:
            >>> graph.add_function("2*x^2 for x in {-3,3} using color:blue, width:3")

        Perl Source: PGgraphmacros.pl add_function method
        """
        # Parse the function string and add to functions list
        # For now, store the raw string/callable - actual rendering would parse this
        func_obj = {'type': 'function',
                    'spec': args[0] if args else '', 'options': kwargs}
        self.functions.append(func_obj)

    def add_vectorfield(self, *args, **kwargs) -> None:
        """
        Add a vector field to the graph.

        Args:
            *args: Vector field specification (string or function)
            **kwargs: Additional options

        Perl Source: PGgraphmacros.pl add_vectorfield method
        """
        field_obj = {'type': 'vectorfield', 'spec': args, 'options': kwargs}
        self.functions.append(field_obj)

    def add_dataset(self, *data, **kwargs) -> None:
        """
        Add a dataset (scatter plot points) to the graph.

        Args:
            *data: Data points (can be list of [x,y] pairs or separate x, y arrays)
            **kwargs: Options like color, style, etc.

        Perl Source: PGgraphmacros.pl add_dataset method
        """
        dataset_obj = {'type': 'dataset', 'data': data, 'options': kwargs}
        self.functions.append(dataset_obj)

    def moveTo(self, x: float, y: float) -> None:
        """Move drawing cursor to position (x, y) without drawing."""
        self._cursor_pos = (x, y)

    def lineTo(self, x: float, y: float, **kwargs) -> None:
        """Draw line from current position to (x, y)."""
        line_obj = {'type': 'line', 'from': getattr(self, '_cursor_pos', (0, 0)),
                    'to': (x, y), 'options': kwargs}
        self.functions.append(line_obj)
        self._cursor_pos = (x, y)

    def arrowTo(self, x: float, y: float, **kwargs) -> None:
        """Draw arrow from current position to (x, y)."""
        arrow_obj = {'type': 'arrow', 'from': getattr(self, '_cursor_pos', (0, 0)),
                     'to': (x, y), 'options': kwargs}
        self.functions.append(arrow_obj)
        self._cursor_pos = (x, y)

    def __repr__(self) -> str:
        """Return string representation of graph."""
        return f"WWPlot({self.width}x{self.height}, bounds=[{self.xmin},{self.ymin},{self.xmax},{self.ymax}])"


class Label:
    """
    Text label for placement on a graph.

    Attributes:
        x, y: Position on graph in real coordinates
        text: Label text content
        color: Color of label text
        horizontal_align: Horizontal alignment (left, center, right)
        vertical_align: Vertical alignment (top, middle, bottom)
    """

    def __init__(self, x: float, y: float, text: str, color: str = 'black',
                 horizontal_align: str = 'center', vertical_align: str = 'middle'):
        """
        Create a label for the graph.

        Args:
            x: x-coordinate position
            y: y-coordinate position
            text: Label text
            color: Text color (default 'black')
            horizontal_align: Alignment left/center/right
            vertical_align: Alignment top/middle/bottom
        """
        self.x = x
        self.y = y
        self.text = str(text)
        self.color = color
        self.horizontal_align = horizontal_align
        self.vertical_align = vertical_align


class Fun:
    """
    Function object for plotting on a graph.

    Wraps a Python function for plotting with configurable colors, weights, and domain.
    """

    def __init__(self, func, graph: WWPlot):
        """
        Create a plottable function.

        Args:
            func: Python function to plot (takes x, returns y)
            graph: Graph object to associate with
        """
        self.func = func
        self.graph = graph
        self.line_color = 'black'
        self.line_weight = 2
        self.domain_min: Optional[float] = None
        self.domain_max: Optional[float] = None

    def color(self, color: str) -> 'Fun':
        """Set the line color."""
        self.line_color = color
        return self

    def weight(self, weight: int) -> 'Fun':
        """Set the line weight (width in pixels)."""
        self.line_weight = weight
        return self

    def domain(self, min_val: float, max_val: float) -> 'Fun':
        """
        Restrict the domain where this function is plotted.

        Args:
            min_val: Minimum x-value
            max_val: Maximum x-value
        """
        self.domain_min = min_val
        self.domain_max = max_val
        return self


def init_graph(xmin: float = -10, xmax: float = 10,
               ymin: float = -10, ymax: float = 10,
               width: int = 200, height: int = 200, **kwargs) -> WWPlot:
    """
    Initialize a graph canvas with specified bounds and size.

    This is the main function for creating graphs in WeBWorK problems.
    It creates a WWPlot object with the given bounds and dimensions.

    Args:
        xmin: Minimum x-coordinate (default -10)
        xmax: Maximum x-coordinate (default 10)
        ymin: Minimum y-coordinate (default -10)
        ymax: Maximum y-coordinate (default 10)
        width: Canvas width in pixels (default 200)
        height: Canvas height in pixels (default 200)
        **kwargs: Additional options (axes, grid, ticks, xlabel, ylabel, etc.)

    Returns:
        WWPlot graph object

    Example:
        >>> graph = init_graph(-5, 5, -5, 5, width=400, height=400)
        >>> graph.h_axis(0, 'black')
        >>> graph.v_axis(0, 'black')

    Perl Source: macros/graph/PGgraphmacros.pl init_graph function
    """
    graph = WWPlot(width=width, height=height)
    graph.xmin = xmin
    graph.xmax = xmax
    graph.ymin = ymin
    graph.ymax = ymax

    # Handle additional options
    if 'axes' in kwargs:
        axes = kwargs['axes']
        if isinstance(axes, list) and len(axes) >= 2:
            graph.h_axis(axes[1])
            graph.v_axis(axes[0])

    if 'grid' in kwargs:
        grid = kwargs['grid']
        if isinstance(grid, list) and len(grid) >= 2:
            # grid format: [xstep, ystep] or [xstep, ystep, color]
            # Add grid support if needed
            pass

    # Handle xlabel/ylabel - store as attributes for rendering
    if 'xlabel' in kwargs:
        graph.xlabel = kwargs['xlabel']
    if 'ylabel' in kwargs:
        graph.ylabel = kwargs['ylabel']

    # Handle tick deltas - store for rendering
    if 'xtick_delta' in kwargs:
        graph.xtick_delta = kwargs['xtick_delta']
    if 'ytick_delta' in kwargs:
        graph.ytick_delta = kwargs['ytick_delta']

    return graph


def add_functions(graph: WWPlot, *functions, **kwargs) -> None:
    """
    Add one or more functions to a graph.

    Functions can be Python callables or string expressions.
    Each function is plotted over the graph's x-range.

    Args:
        graph: WWPlot graph object to add functions to
        *functions: Functions to plot (callable or Fun objects)
        **kwargs: Options like color, weight, domain

    Example:
        >>> graph = init_graph(-5, 5, -5, 5)
        >>> add_functions(graph, lambda x: x**2, lambda x: x**3)

    Perl Source: macros/graph/PGgraphmacros.pl add_functions
    """
    for func in functions:
        if callable(func):
            fun_obj = Fun(func, graph)
            graph.functions.append(fun_obj)
        elif isinstance(func, Fun):
            graph.functions.append(func)


def Plot(*args, **kwargs):
    """
    Convenience wrapper for plotting - creates graph with functions.

    This is a simplified interface that combines init_graph and function plotting.

    Args:
        *args: Arguments passed to init_graph
        **kwargs: Keyword arguments passed to init_graph

    Returns:
        WWPlot graph object

    Example:
        >>> plot = Plot(-5, 5, -5, 5)

    Perl Source: PGgraphmacros.pl Plot function
    """
    return init_graph(*args, **kwargs)


__all__ = [
    'WWPlot',
    'Label',
    'Fun',
    'init_graph',
    'add_functions',
    'Plot',
]
