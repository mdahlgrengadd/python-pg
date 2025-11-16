"""
PG Utility Functions - Mathematical and helper functions.

Common utility functions used in PG problems.
Reference: PGauxiliaryFunctions.pl

This module re-exports functions from pg_auxiliary_functions for backwards compatibility.
"""

from typing import Any

# Import all duplicated functions from the canonical implementation
from .pg_auxiliary_functions import (
    gcf,
    gcd,
    lcm,
    reduce as reduce_fraction,  # Alias reduce to reduce_fraction
    step,
    max as max_number,  # Alias max to max_number
    min as min_number,  # Alias min to min_number
    factorial as fact,
    isPrime,
    preformat,
    random_coprime,
    random_pairwise_coprime,
)

# Additional utility functions (not direct duplicates)
import math
from typing import Any


def C(n: int, k: int) -> int:
    """
    Binomial coefficient: n choose k.
    
    Args:
        n: Total items
        k: Items to choose
        
    Returns:
        Number of ways to choose k items from n
        
    Examples:
        >>> C(5, 2)
        10
        >>> C(10, 3)
        120
        
    Reference: PGauxiliaryFunctions.pl::C
    """
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def P(n: int, k: int) -> int:
    """
    Permutations: n permute k.
    
    Args:
        n: Total items
        k: Items to arrange
        
    Returns:
        Number of ways to arrange k items from n
        
    Examples:
        >>> P(5, 2)
        20
        >>> P(10, 3)
        720
        
    Reference: PGauxiliaryFunctions.pl::P
    """
    if k < 0 or k > n:
        return 0
    return math.perm(n, k)


def lex_sort(*items: str) -> list[str]:
    """
    Lexicographic (alphabetical) sort.
    
    Args:
        *items: Strings to sort
        
    Returns:
        Sorted list of strings
        
    Examples:
        >>> lex_sort("zebra", "apple", "banana")
        ['apple', 'banana', 'zebra']
        
    Reference: PGauxiliaryFunctions.pl::lex_sort
    """
    return sorted(items)


def num_sort(*numbers: float) -> list[float]:
    """
    Numerical sort.
    
    Args:
        *numbers: Numbers to sort
        
    Returns:
        Sorted list of numbers
        
    Examples:
        >>> num_sort(5, 2, 8, 1)
        [1, 2, 5, 8]
        
    Reference: PGauxiliaryFunctions.pl::num_sort
    """
    return sorted(numbers)


def uniq(*items: Any) -> list[Any]:
    """
    Remove duplicates from list while preserving order.
    
    Args:
        *items: Items (possibly with duplicates)
        
    Returns:
        List with duplicates removed
        
    Examples:
        >>> uniq(1, 2, 2, 3, 1, 4)
        [1, 2, 3, 4]
        
    Reference: PGauxiliaryFunctions.pl::uniq
    """
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def sgn(x: float) -> int:
    """
    Sign function.
    
    Args:
        x: Number
        
    Returns:
        -1 if x < 0, 0 if x == 0, 1 if x > 0
        
    Examples:
        >>> sgn(-5)
        -1
        >>> sgn(0)
        0
        >>> sgn(3.7)
        1
        
    Reference: PGauxiliaryFunctions.pl::sgn
    """
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0


# Export all functions for backwards compatibility
__all__ = [
    "gcf",
    "gcd",
    "lcm",
    "reduce_fraction",
    "sgn",
    "max_number",
    "min_number",
    "step",
    "fact",
    "C",
    "P",
    "lex_sort",
    "num_sort",
    "uniq",
    "isPrime",
    "preformat",
    "random_coprime",
    "random_pairwise_coprime",
]

