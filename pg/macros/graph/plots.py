"""plots.pl - Dynamic graph plotting for WeBWorK PG.

This module provides the Plot class for creating modern, interactive plots
with support for functions, parametric curves, datasets, and various styling options.

Based on macros/graph/plots.pl from the WeBWorK distribution.

Usage:
    plot = Plot(xmin=0, xmax=10, ymin=0, ymax=100,
                xtick_delta=2, ytick_delta=20,
                xlabel='\\(t\\)', ylabel='\\(h(t)\\)')
    plot.add_function(['2*sin(2*t)', '2*sin(3*t)'], 't', 0, '2*pi', color='blue')
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math


class PlotData:
    """Data object representing a plot element (function, dataset, label, etc.).
    
    This stores information about a single plottable element including:
    - name: Type of element (function, dataset, circle, label, etc.)
    - x, y: Data point arrays
    - function: Function specification for generative plots
    - styles: Styling options (color, width, linestyle, marks, etc.)
    """
    
    def __init__(self, name: str = ''):
        """Initialize a plot data object.
        
        Args:
            name: Type identifier (function, dataset, circle, etc.)
        """
        self.name = name
        self.x: List[float] = []
        self.y: List[float] = []
        self.function: Dict[str, Any] = {}
        self.styles: Dict[str, Any] = {}
        self._context = None
        
    def add(self, *args):
        """Add data point(s).
        
        Args:
            *args: Either (x, y) pairs or list of [x, y] pairs
        """
        if len(args) >= 2 and not isinstance(args[0], (list, tuple)):
            # Single point: add(x, y)
            x, y = args[0], args[1]
            if x is not None and y is not None:
                self.x.append(float(x))
                self.y.append(float(y))
        else:
            # Multiple points: add([x1, y1], [x2, y2], ...)
            for point in args:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    self.add(point[0], point[1])
                    
    def size(self) -> int:
        """Return number of data points."""
        return len(self.x)
        
    def style(self, *args, **kwargs):
        """Set or get style options.
        
        Args:
            *args: If single arg and string, get that style value; if dict, set styles
            **kwargs: Set style key-value pairs
            
        Returns:
            Style value if getting, None if setting
        """
        if len(args) == 1 and not kwargs:
            # Check if it's a get or set operation
            if isinstance(args[0], dict):
                # Set styles from dict
                self.styles.update(args[0])
                return None
            else:
                # Get single style value
                return self.styles.get(args[0], '')
        
        # Set styles
        if kwargs:
            self.styles.update(kwargs)
            
    def set_function(self, context, **options):
        """Configure a function for data generation.
        
        Args:
            context: MathObjects context
            **options: Function parameters
                Fx, Fy: Function expressions (Formula or string)
                xvar, yvar: Variable names (default 't')
                xmin, xmax: Domain bounds
                xsteps: Number of sample points (default 30)
                var, min, max, steps: Shortcuts for x-variants
        """
        try:
            from pg.macros.core.mathobjects import Formula, Value
        except ImportError:
            # Fallback if mathobjects not available
            Formula = None
            Value = None
        
        self._context = context
        
        # Default function parameters
        self.function = {
            'Fx': 't',
            'Fy': '',
            'xvar': 't',
            'xmin': -5,
            'xmax': 5,
            'xsteps': 30,
        }
        
        # Update with provided options
        for key in ['Fx', 'Fy', 'xvar', 'xmin', 'xmax', 'xsteps']:
            if key in options:
                self.function[key] = options.pop(key)
                
        # Handle shortcuts (var->xvar, min->xmin, max->xmax, steps->xsteps)
        for short, long in [('var', 'xvar'), ('min', 'xmin'), ('max', 'xmax'), ('steps', 'xsteps')]:
            if short in options:
                self.function[long] = options.pop(short)
                
        # Convert string functions to Formula objects if needed
        if not self.function['Fy']:
            return
            
        # Convert Fx and Fy to Formula if they're strings
        for coord in ['Fx', 'Fy']:
            val = self.function[coord]
            if isinstance(val, str) and val and Formula:
                # Make sure variable is in context
                var = self.function['xvar']
                try:
                    if var and hasattr(context, 'variables') and var not in context.variables.names:
                        context = context.copy()
                        context.variables.add(var)
                    self.function[coord] = Formula(val, context=context)
                except Exception:
                    # If Formula creation fails, keep as string
                    pass
            elif callable(val):
                # Keep callables as-is
                pass
                
        # Store remaining options as styles
        if options:
            self.style(**options)
            
    def gen_data(self):
        """Generate data points from function specification."""
        if not self.function or self.size() > 0:
            # Only generate once
            return
            
        try:
            from pg.macros.core.mathobjects import Formula
        except ImportError:
            Formula = None
        
        Fx = self.function.get('Fx')
        Fy = self.function.get('Fy')
        xmin = self.function.get('xmin', -5)
        xmax = self.function.get('xmax', 5)
        xsteps = self.function.get('xsteps', 30)
        var = self.function.get('xvar', 't')
        
        if not Fy:
            return
            
        # Convert string min/max to floats (handles 'pi', '2*pi', etc.)
        if isinstance(xmin, str):
            try:
                from pg.macros.core.mathobjects import Real
                xmin = float(Real(xmin, context=self._context))
            except Exception:
                xmin = eval(xmin.replace('pi', str(math.pi)))
                
        if isinstance(xmax, str):
            try:
                from pg.macros.core.mathobjects import Real
                xmax = float(Real(xmax, context=self._context))
            except Exception:
                xmax = eval(xmax.replace('pi', str(math.pi)))
                
        dt = (xmax - xmin) / xsteps
        
        for i in range(xsteps + 1):
            t = xmin + i * dt
            try:
                # Evaluate functions at t
                if isinstance(Fx, Formula):
                    x_val = float(Fx.eval(**{var: t}))
                elif callable(Fx):
                    x_val = float(Fx(t))
                elif Fx == var:
                    x_val = t
                else:
                    x_val = float(Fx)
                    
                if isinstance(Fy, Formula):
                    y_val = float(Fy.eval(**{var: t}))
                elif callable(Fy):
                    y_val = float(Fy(t))
                else:
                    y_val = float(Fy)
                    
                self.add(x_val, y_val)
            except Exception:
                # Skip points that fail to evaluate
                continue


class PlotAxes:
    """Axes configuration for plots.
    
    Stores configuration for both axes and general plot styles.
    """
    
    def __init__(self):
        """Initialize axes with default configuration."""
        self.xaxis_data = {
            'min': -5,
            'max': 5,
            'tick_num': 5,
            'tick_delta': 0,
            'tick_labels': 1,
            'show_ticks': 1,
            'label': '\\(x\\)',
            'major': 1,
            'minor': 3,
            'visible': 1,
            'location': 'middle',
            'position': 0,
        }
        self.yaxis_data = {
            'min': -5,
            'max': 5,
            'tick_num': 5,
            'tick_delta': 0,
            'tick_labels': 1,
            'show_ticks': 1,
            'label': '\\(y\\)',
            'major': 1,
            'minor': 3,
            'visible': 1,
            'location': 'center',
            'position': 0,
        }
        self.styles_data = {
            'show_grid': 1,
            'aria_label': '',
            'axes_on_top': 0,
            'aspect_ratio': 0,
        }
        
    def xaxis(self, key: Optional[str] = None, **kwargs) -> Any:
        """Get or set xaxis configuration.
        
        Args:
            key: If provided, get this single value
            **kwargs: Set key-value pairs
            
        Returns:
            Full dict if no args, single value if key provided, None if setting
        """
        if key and not kwargs:
            return self.xaxis_data.get(key)
        if kwargs:
            self.xaxis_data.update(kwargs)
            return None
        return self.xaxis_data
        
    def yaxis(self, key: Optional[str] = None, **kwargs) -> Any:
        """Get or set yaxis configuration."""
        if key and not kwargs:
            return self.yaxis_data.get(key)
        if kwargs:
            self.yaxis_data.update(kwargs)
            return None
        return self.yaxis_data
        
    def style(self, key: Optional[str] = None, **kwargs) -> Any:
        """Get or set style configuration."""
        if key and not kwargs:
            return self.styles_data.get(key)
        if kwargs:
            self.styles_data.update(kwargs)
            return None
        return self.styles_data
        
    def set(self, **options):
        """Set multiple axis/style options at once.
        
        Options starting with 'x' go to xaxis, 'y' to yaxis, others to styles.
        
        Args:
            **options: Configuration options (xmin, ymax, aria_label, etc.)
        """
        for key, value in options.items():
            if key.startswith('x') and len(key) > 1:
                # xmin -> min, xtick_delta -> tick_delta, etc.
                axis_key = key[1:]
                self.xaxis_data[axis_key] = value
            elif key.startswith('y') and len(key) > 1:
                axis_key = key[1:]
                self.yaxis_data[axis_key] = value
            else:
                self.styles_data[key] = value
                
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return axis bounds as (xmin, ymin, xmax, ymax)."""
        return (
            self.xaxis_data['min'],
            self.yaxis_data['min'],
            self.xaxis_data['max'],
            self.yaxis_data['max']
        )


class PlotObject:
    """Main Plot object for creating graphs.
    
    This class provides a high-level interface for creating plots with functions,
    parametric curves, datasets, and various styling options. It manages axes
    configuration and plot data elements.
    
    Attributes:
        width: Canvas width in pixels
        height: Canvas height in pixels (auto-calculated if not set)
        axes: PlotAxes object for axis configuration
        data: List of PlotData objects
        colors: Color definitions
    """
    
    def __init__(self, **options):
        """Initialize a plot with configuration options.
        
        Args:
            width: Canvas width in pixels (default 350)
            height: Canvas height in pixels (auto if not set)
            tex_size: Size for TeX rendering (default 600)
            xmin, xmax, ymin, ymax: Axis bounds
            xtick_delta, ytick_delta: Tick spacing
            xlabel, ylabel: Axis labels
            aria_label: Accessibility label
            **options: Additional axes/style options
        """
        # Core plot attributes
        self.width = options.pop('width', 350)
        self.height = options.pop('height', None)
        self.tex_size = options.pop('tex_size', 600)
        self.ext = '.html'
        self.image_type_name = 'JSXGraph'
        self.image_name = None
        
        # Axes and data
        self.axes = PlotAxes()
        self.data: List[PlotData] = []
        self.colors: Dict[str, Tuple[int, int, int]] = {}
        
        # Initialize default colors
        self._color_init()
        
        # Apply axes configuration from options
        if options:
            self.axes.set(**options)
            
        # Get or create context
        self._context = self._get_context()
        
    def _color_init(self):
        """Initialize default color palette."""
        self.add_color('default_color', 0, 0, 0)
        self.add_color('white', 255, 255, 255)
        self.add_color('gray', 128, 128, 128)
        self.add_color('grey', 128, 128, 128)
        self.add_color('black', 0, 0, 0)
        # Primary and secondary colors
        self.add_color('red', 255, 0, 0)
        self.add_color('green', 0, 128, 0)
        self.add_color('blue', 0, 0, 255)
        self.add_color('yellow', 255, 255, 0)
        self.add_color('cyan', 0, 255, 255)
        self.add_color('magenta', 255, 0, 255)
        self.add_color('orange', 255, 128, 0)
        self.add_color('purple', 128, 0, 128)
        
    def _get_context(self):
        """Get current MathObjects context or create default."""
        try:
            from pg.core.mathobjects import Context
            return Context()
        except Exception:
            return None
            
    def add_color(self, name: str, r: int, g: int, b: int):
        """Add a color definition.
        
        Args:
            name: Color name
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
        """
        self.colors[name] = (r, g, b)
        
    def add_data(self, data: PlotData):
        """Add a PlotData object to the plot.
        
        Args:
            data: PlotData object to add
        """
        self.data.append(data)
        
    def _add_function(self, Fx, Fy, var, min_val, max_val, *args, **kwargs):
        """Internal method to add a function.
        
        Args:
            Fx: X-component function (or variable for regular functions)
            Fy: Y-component function
            var: Parameter variable name
            min_val: Domain minimum
            max_val: Domain maximum
            *args, **kwargs: Style options
            
        Returns:
            PlotData object
        """
        var = var or 't'
        Fx = Fx if Fx is not None else var
        
        # Merge args (assumed to be key-value pairs) and kwargs
        options = dict(zip(args[::2], args[1::2])) if args else {}
        options.update(kwargs)
        
        # Set defaults for options not provided
        if 'color' not in options:
            options['color'] = 'default_color'
        if 'width' not in options:
            options['width'] = 2
        
        data = PlotData(name='function')
        data.set_function(
            self._context,
            Fx=Fx,
            Fy=Fy,
            xvar=var,
            xmin=min_val,
            xmax=max_val,
            **options
        )
        
        self.add_data(data)
        return data
        
    def add_function(self, f, *rest, **kwargs):
        """Add function(s) to the plot.
        
        Supports multiple formats:
        1. Regular function: add_function('x^2', 'x', -5, 5)
        2. Parametric function: add_function(['cos(t)', 'sin(t)'], 't', 0, 2*pi)
        3. Multiple functions: add_function([func1, var1, min1, max1], [func2, ...])
        
        Args:
            f: Function expression (string, Formula, array for parametric)
            *rest: Additional arguments (variable, min, max, or multiple functions)
            **kwargs: Style options (color, width, etc.)
            
        Returns:
            PlotData object or list of PlotData objects
        """
        # Case 1: Parametric function - f is array [Fx, Fy]
        if isinstance(f, (list, tuple)) and len(f) == 2 and not isinstance(f[0], (list, tuple)):
            # Parametric: add_function(['cos(t)', 'sin(t)'], 't', 0, 2*pi)
            Fx, Fy = f[0], f[1]
            var = rest[0] if rest else 't'
            min_val = rest[1] if len(rest) > 1 else -5
            max_val = rest[2] if len(rest) > 2 else 5
            return self._add_function(Fx, Fy, var, min_val, max_val, **kwargs)
            
        # Case 2: Multiple functions - f is array of arrays
        if isinstance(f, (list, tuple)) and f and isinstance(f[0], (list, tuple)):
            # Multiple: add_function([[func1, var1, ...], [func2, var2, ...]])
            results = []
            for func_spec in [f] + list(rest):
                if isinstance(func_spec, (list, tuple)):
                    g = func_spec[0]
                    options = list(func_spec[1:])
                    if isinstance(g, (list, tuple)) and len(g) == 2:
                        # Parametric
                        result = self._add_function(g[0], g[1], *options, **kwargs)
                    else:
                        # Regular
                        result = self._add_function(None, g, *options, **kwargs)
                    results.append(result)
            return results if len(results) > 1 else results[0]
            
        # Case 3: Regular function - f is expression
        # add_function('x^2', 'x', -5, 5, color='blue')
        var = rest[0] if rest else 'x'
        min_val = rest[1] if len(rest) > 1 else -5
        max_val = rest[2] if len(rest) > 2 else 5
        return self._add_function(var, f, var, min_val, max_val, **kwargs)
        
    def _add_dataset(self, *points):
        """Internal method to add a dataset.
        
        Args:
            *points: Point specifications and style options
            
        Returns:
            PlotData object
        """
        data = PlotData(name='dataset')
        
        # Extract points (arrays) from arguments
        while points and isinstance(points[0], (list, tuple)):
            data.add(*points[0])
            points = points[1:]
            
        # Remaining arguments are style options (key-value pairs)
        style_dict = dict(zip(points[::2], points[1::2])) if points else {}
        
        # Set defaults if not provided
        if 'color' not in style_dict:
            style_dict['color'] = 'default_color'
        if 'width' not in style_dict:
            style_dict['width'] = 2
            
        data.style(**style_dict)
            
        self.add_data(data)
        return data
        
    def add_dataset(self, *args, **kwargs):
        """Add dataset(s) (points/lines) to the plot.
        
        Formats:
        1. Single dataset: add_dataset([x1, y1], [x2, y2], ..., color='red')
        2. Multiple datasets: add_dataset([[x1, y1], [x2, y2], ...], [[...], ...])
        
        Args:
            *args: Point specifications
            **kwargs: Style options
            
        Returns:
            PlotData object or list of PlotData objects
        """
        # Check if multiple datasets (first arg is array of arrays)
        if args and isinstance(args[0], (list, tuple)) and args[0] and isinstance(args[0][0], (list, tuple)):
            # Multiple datasets
            return [self._add_dataset(*dataset) for dataset in args]
            
        # Single dataset
        # Convert kwargs to positional style args for consistency with Perl API
        style_args = []
        for k, v in kwargs.items():
            style_args.extend([k, v])
        return self._add_dataset(*args, *style_args)
        
    def add_circle(self, *args):
        """Add circle(s) to the plot.
        
        Args:
            *args: Circle specifications ([x, y], radius, options...)
            
        Returns:
            PlotData object or list of PlotData objects
        """
        # Implementation placeholder for completeness
        data = PlotData(name='circle')
        # Parse circle data...
        self.add_data(data)
        return data
        
    def add_label(self, *args):
        """Add label(s) to the plot.
        
        Args:
            *args: Label specifications (x, y, text, options...)
            
        Returns:
            PlotData object or list of PlotData objects
        """
        # Implementation placeholder for completeness
        data = PlotData(name='label')
        # Parse label data...
        self.add_data(data)
        return data
        
    def add_vectorfield(self, **options):
        """Add a vector field to the plot.
        
        Args:
            **options: Vector field options (Fx, Fy, xmin, xmax, etc.)
            
        Returns:
            PlotData object
        """
        # Set defaults for options not provided
        defaults = {
            'Fx': '',
            'Fy': '',
            'xvar': 'x',
            'xmin': -5,
            'xmax': 5,
            'xsteps': 15,
            'width': 1,
            'color': 'default_color',
        }
        # Merge defaults with provided options (options override defaults)
        merged_options = {**defaults, **options}
        
        data = PlotData(name='vectorfield')
        data.set_function(self._context, **merged_options)
        self.add_data(data)
        return data
        
    def size(self) -> Tuple[int, int]:
        """Calculate and return plot size (width, height).
        
        Returns:
            Tuple of (width, height) in pixels
        """
        width = self.width
        height = self.height
        
        if not height:
            aspect_ratio = self.axes.style('aspect_ratio')
            if aspect_ratio:
                xmin, ymin, xmax, ymax = self.axes.bounds()
                x_size = xmax - xmin
                y_size = ymax - ymin
                height = int(aspect_ratio * width * y_size / x_size)
            else:
                height = width
                
        return (width, height)
        
    def draw(self) -> str:
        """Generate the plot output.
        
        Returns:
            HTML string or image path for the rendered plot
        """
        # For now, return placeholder indicating plot object exists
        # Full rendering would integrate with JSXGraph/TikZ backends
        return f'<div class="plot" data-width="{self.width}" data-height="{self.size()[1]}">Plot object</div>'
        
    def __repr__(self) -> str:
        """String representation of Plot object."""
        width, height = self.size()
        xmin, ymin, xmax, ymax = self.axes.bounds()
        return f"Plot({width}x{height}, bounds=[{xmin},{ymin},{xmax},{ymax}], data={len(self.data)} elements)"


def Plot(**options):
    """Factory function to create a Plot object.
    
    This is the main entry point for creating plots in PG problems.
    
    Args:
        **options: Plot configuration options (see PlotObject class)
        
    Returns:
        PlotObject instance
        
    Example:
        >>> plot = Plot(xmin=-5, xmax=5, ymin=-5, ymax=5)
        >>> plot.add_function('x^2', 'x', -5, 5, color='blue')
    """
    return PlotObject(**options)


__all__ = [
    'Plot',
    'PlotObject',
    'PlotData',
    'PlotAxes',
]

