"""Graph and plot macros for WeBWorK PG."""


class Plot:
    """A plot object for creating graphs."""

    def __init__(self, **kwargs):
        """Initialize a plot with given parameters.

        Args:
            xmin: Minimum x value
            xmax: Maximum x value
            ymin: Minimum y value
            ymax: Maximum y value
            aria_label: Accessibility label for the plot
            **kwargs: Additional plot parameters
        """
        self.xmin = kwargs.get('xmin', 0)
        self.xmax = kwargs.get('xmax', 10)
        self.ymin = kwargs.get('ymin', 0)
        self.ymax = kwargs.get('ymax', 10)
        self.aria_label = kwargs.get('aria_label', '')
        self.datasets = []
        self.functions = []

    def add_function(self, formula, **kwargs):
        """Add a function to the plot.

        Args:
            formula: A Formula object or string
            **kwargs: Additional options for the function
        """
        self.functions.append({'formula': formula, 'options': kwargs})

    def add_dataset(self, *args, **kwargs):
        """Add a dataset (points or lines) to the plot.

        Args:
            *args: Points or path data
            **kwargs: Style options (fill, fill_color, fill_opacity, etc.)
        """
        self.datasets.append({'data': args, 'options': kwargs})


def Plot(**kwargs):
    """Factory function to create a Plot object."""
    return Plot(**kwargs)
