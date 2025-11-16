"""
Statistics Utilities for WeBWorK.

This module provides statistical functions for WeBWorK problems,
including linear regression calculations.

Based on PGstatisticsmacros.pl from the Perl WeBWorK distribution.
"""

from typing import Any, List, Optional, Tuple, Union


def sample_correlation(x_values: List[float], y_values: List[float]) -> float:
    """
    Calculate the sample correlation coefficient (Pearson's r).

    Computes the correlation between two sets of values.

    Args:
        x_values: First set of values
        y_values: Second set of values

    Returns:
        The correlation coefficient between -1 and 1

    Perl Source: PGstatisticsmacros.pl sample_correlation function
    """
    import math

    if len(x_values) < 2 or len(y_values) < 2 or len(x_values) != len(y_values):
        return 0.0

    n = len(x_values)
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n

    numerator = sum((x_values[i] - mean_x) * (y_values[i] - mean_y) for i in range(n))
    sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
    sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)

    denominator = math.sqrt(sum_sq_x * sum_sq_y)

    if denominator == 0:
        return 0.0

    return numerator / denominator


def linear_regression(x_values: List[float], y_values: List[float]) -> Tuple[float, float]:
    """
    Calculate linear regression statistics.

    Performs linear regression calculation on data points, returning
    the slope and intercept of the best-fit line.

    Args:
        x_values: Independent variable values
        y_values: Dependent variable values

    Returns:
        Tuple of (slope, intercept) for the regression line

    Example:
        >>> from pg.macros.math.statistics_utils import linear_regression
        >>> slope, intercept = linear_regression([1, 2, 3], [2, 4, 5])
        >>> # Returns (slope=1.5, intercept=0.5) approximately

    Perl Source: PGstatisticsmacros.pl linear_regression function
    """
    import math

    if len(x_values) < 2 or len(y_values) < 2 or len(x_values) != len(y_values):
        return (1, 0)

    n = len(x_values)
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n

    numerator = sum((x_values[i] - mean_x) * (y_values[i] - mean_y) for i in range(n))
    denominator = sum((x - mean_x) ** 2 for x in x_values)

    if denominator == 0:
        return (1, 0)

    slope = numerator / denominator
    intercept = mean_y - slope * mean_x

    return (slope, intercept)


def stats_mean(*values: Any) -> float:
    """
    Calculate the mean (average) of values.
    
    Args:
        *values: Values to average, or a single list/tuple of values
        
    Returns:
        The arithmetic mean of the values
        
    Perl Source: PGstatisticsmacros.pl stats_mean function
    """
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = values[0]
    return sum(values) / len(values) if values else 0.0


def stats_sd(*values: Any) -> float:
    """
    Calculate the standard deviation of values.
    
    Args:
        *values: Values to analyze, or a single list/tuple of values
        
    Returns:
        The sample standard deviation
        
    Perl Source: PGstatisticsmacros.pl stats_sd function
    """
    import math
    
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = values[0]
    
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def stats_SX_SXX(*values: Any) -> Tuple[float, float]:
    """
    Calculate sum of X and sum of X squared.
    
    Args:
        *values: Values to analyze, or a single list/tuple of values
        
    Returns:
        Tuple of (sum_x, sum_x_squared)
        
    Perl Source: PGstatisticsmacros.pl stats_SX_SXX function
    """
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = values[0]
    
    sum_x = sum(values) if values else 0.0
    sum_sq = sum(x ** 2 for x in values) if values else 0.0
    return (sum_x, sum_sq)


__all__ = [
    'sample_correlation',
    'linear_regression',
    'stats_mean',
    'stats_sd',
    'stats_SX_SXX',
]

