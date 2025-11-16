"""Statistics functions for PG problems."""

import math
from typing import List, Tuple, Union


def stats_mean(*values: Union[float, int, List]) -> float:
    """Calculate the mean of a list of numbers.

    Args:
        *values: Numbers to average (can be individual args or a list)

    Returns:
        Mean of the values
    """
    # Flatten if passed as list
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = values[0]

    if not values:
        return 0.0

    return sum(values) / len(values)


def stats_sd(*values: Union[float, int, List]) -> float:
    """Calculate the sample standard deviation of a list of numbers.

    Uses the n-1 denominator (sample standard deviation).

    Args:
        *values: Numbers to calculate standard deviation for

    Returns:
        Sample standard deviation
    """
    # Flatten if passed as list
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = values[0]

    if len(values) < 2:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def stats_SX_SXX(*values: Union[float, int, List]) -> Tuple[float, float]:
    """Calculate sum and sum of squares of a list of numbers.

    Args:
        *values: Numbers to process

    Returns:
        Tuple of (sum, sum_of_squares)
    """
    # Flatten if passed as list
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = values[0]

    if not values:
        return (0.0, 0.0)

    sum_x = sum(values)
    sum_sq = sum(x ** 2 for x in values)

    return (sum_x, sum_sq)
