"""Auxiliary Math Functions for WeBWorK.

This module provides utility functions including rounding, polynomial formatting,
random person generation, and 3D point/vector generation.

Based on PGauxiliaryFunctions.pl from the WeBWorK distribution.
"""

import random
import math
from typing import List, Optional, Union, Any


def Round(value: Union[int, float, str], decimals: int = 0) -> float:
    """
    Round a number to a specified number of decimal places.
    
    Args:
        value: Number to round (int, float, or string representation)
        decimals: Number of decimal places (default 0)
        
    Returns:
        Rounded value as a float
        
    Example:
        >>> Round(3.14159, 2)
        3.14
        >>> Round(2.5)
        2.0
    
    Perl Source: PGauxiliaryFunctions.pl Round function
    """
    try:
        return round(float(value), int(decimals))
    except (ValueError, TypeError):
        return 0.0


def nicestring(coeffs: List[Union[int, float]], vars: Optional[List[str]] = None) -> str:
    """
    Format a polynomial as a nice string.
    
    Converts a list of coefficients to a polynomial string like "3x^2 + 2x - 5"
    
    Args:
        coeffs: List of coefficients (e.g., [1, 2, -3] for x^2 + 2x - 3)
        vars: List of variable names (default: ['x'])
        
    Returns:
        String representation of the polynomial
        
    Example:
        >>> nicestring([1, 2, -3])
        'x^2 + 2x - 3'
        >>> nicestring([3, 0, 5], ['a'])
        '3a^2 + 5'
    
    Perl Source: PGauxiliaryFunctions.pl nicestring function
    """
    if vars is None:
        vars = ['x']
    
    if not coeffs:
        return '0'
    
    terms = []
    
    for i, coeff in enumerate(coeffs):
        if coeff == 0:
            continue
        
        var = vars[i] if i < len(vars) else ''
        power = len(coeffs) - 1 - i
        
        # Build the term
        if power == 0:
            # Constant term
            terms.append(str(int(coeff) if coeff == int(coeff) else coeff))
        elif power == 1:
            # Linear term
            if coeff == 1:
                terms.append(var)
            elif coeff == -1:
                terms.append(f"-{var}")
            else:
                terms.append(f"{int(coeff) if coeff == int(coeff) else coeff}{var}")
        else:
            # Higher power terms
            if coeff == 1:
                terms.append(f"{var}^{power}")
            elif coeff == -1:
                terms.append(f"-{var}^{power}")
            else:
                terms.append(f"{int(coeff) if coeff == int(coeff) else coeff}{var}^{power}")
    
    if not terms:
        return '0'
    
    # Join terms with appropriate signs
    result = terms[0]
    for term in terms[1:]:
        if term.startswith('-'):
            result += f" - {term[1:]}"
        else:
            result += f" + {term}"
    
    return result


def randomPerson(n: int = 1, **kwargs) -> Union['Person', List['Person']]:
    """
    Generate random person names with optional pronouns.
    
    Args:
        n: Number of persons to generate (default 1)
        **kwargs: Additional options
        
    Returns:
        Single Person object if n=1, list of Person objects otherwise
        
    Example:
        >>> person = randomPerson()
        >>> str(person)
        'Alice Smith'
        >>> persons = randomPerson(3)
        >>> len(persons)
        3
    
    Perl Source: PGauxiliaryFunctions.pl randomPerson function
    """
    class Person:
        """Random person with name."""
        def __init__(self, first: str, last: str):
            self._first = first
            self._last = last
        
        def name(self) -> str:
            """Get full name."""
            return f"{self._first} {self._last}"
        
        def __str__(self) -> str:
            """String representation."""
            return self.name()
    
    first_names = ['Alice', 'Bob', 'Carol', 'David', 'Eve', 'Frank', 'Grace', 'Henry']
    last_names = ['Smith', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor']
    
    if n == 1:
        return Person(random.choice(first_names), random.choice(last_names))
    else:
        return [Person(random.choice(first_names), random.choice(last_names)) for _ in range(n)]


def non_zero_point3D(*args, **kwargs) -> List[float]:
    """
    Generate a random non-zero 3D point.
    
    All coordinates are non-zero integers in range [-5, 5].
    
    Args:
        *args: Optional range arguments (unused in basic version)
        **kwargs: Additional options
        
    Returns:
        List [x, y, z] with no zero coordinates
        
    Example:
        >>> point = non_zero_point3D()
        >>> len(point)
        3
        >>> all(p != 0 for p in point)
        True
    
    Perl Source: PGauxiliaryFunctions.pl non_zero_point3D function
    """
    def non_zero_random(min_val: int = -5, max_val: int = 5) -> int:
        """Generate random non-zero integer."""
        while True:
            val = random.randint(min_val, max_val)
            if val != 0:
                return val
    
    return [non_zero_random(), non_zero_random(), non_zero_random()]


def non_zero_vector3D(*args, **kwargs):
    """
    Generate a random non-zero 3D vector as a Vector MathObject.

    All components are non-zero integers in range (or specified range).

    Args:
        *args: Optional range arguments (low, high, step)
               If provided: (low, high, step) - generates values in [low, high] by step
               If not provided: uses range [-5, 5] by 1
        **kwargs: Additional options (seed, rng, etc.)

    Returns:
        Vector object with 3 components [x, y, z]

    Example:
        >>> vector = non_zero_vector3D()
        >>> vector.components
        [x, y, z]
        >>> all(v != 0 for v in vector.value)
        True

    Perl Source: PGauxiliaryFunctions.pl non_zero_vector3D function
    """
    from pg.math.geometric import Vector

    # Generate the component list using the point function
    components = non_zero_point3D(*args, **kwargs)

    # Wrap in Vector MathObject and return
    result = Vector(components)
    return result


__all__ = [
    'Round',
    'nicestring',
    'randomPerson',
    'non_zero_point3D',
    'non_zero_vector3D',
]

