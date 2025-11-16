"""
PGauxiliaryFunctions.pl - Auxiliary mathematical functions for PG

Reference: macros/core/PGauxiliaryFunctions.pl (600+ lines)
"""

import math
from typing import Union, List, Tuple, Any
from functools import reduce as _reduce
import random as _random


def step(x: Union[int, float]) -> int:
    """
    Heaviside/step function.

    Returns 1 if x >= 0, else 0.

    Reference: PGauxiliaryFunctions.pl::step

    Example:
        step(3.14159) → 1
        step(-1) → 0
    """
    return 1 if x >= 0 else 0


def ceil(x: Union[int, float]) -> int:
    """
    Ceiling function - rounds up to nearest integer.

    Reference: PGauxiliaryFunctions.pl::ceil

    Example:
        ceil(3.14159) → 4
        ceil(-9.75) → -9
    """
    return math.ceil(x)


def floor(x: Union[int, float]) -> int:
    """
    Floor function - rounds down to nearest integer.

    Reference: PGauxiliaryFunctions.pl::floor

    Example:
        floor(3.14159) → 3
        floor(-9.75) → -10
    """
    return math.floor(x)


def max(*args: Union[int, float]) -> Union[int, float]:
    """
    Returns maximum value from arguments.

    Reference: PGauxiliaryFunctions.pl::max

    Example:
        max(1, 2, 3, 4, 5, 6, 7) → 7
    """
    if not args:
        raise ValueError("max expected at least 1 argument, got 0")
    return _reduce(lambda a, b: a if a > b else b, args)


def min(*args: Union[int, float]) -> Union[int, float]:
    """
    Returns minimum value from arguments.

    Reference: PGauxiliaryFunctions.pl::min

    Example:
        min(1, 2, 3, 4, 5, 6, 7) → 1
    """
    if not args:
        raise ValueError("min expected at least 1 argument, got 0")
    return _reduce(lambda a, b: a if a < b else b, args)


# Store reference to built-in round before defining our own
_builtin_round = __builtins__['round'] if isinstance(__builtins__, dict) else __builtins__.round


def round_to(x: Union[int, float], n: int = 0) -> Union[int, float]:
    """
    Round to n decimal places (or nearest integer if n=0).

    Reference: PGauxiliaryFunctions.pl::Round

    Example:
        round_to(1.789, 2) → 1.79
        round_to(3.14159) → 3
    """
    if n == 0:
        # Standard banker's rounding in Python
        return int(_builtin_round(x))
    else:
        # Round to n decimal places
        factor = 10 ** n
        return _builtin_round(x * factor) / factor


def round(x: Union[int, float]) -> int:
    """
    Round to nearest integer.

    Reference: PGauxiliaryFunctions.pl::round

    Example:
        round(3.14159) → 3
        round(3.6) → 4
    """
    return int(_builtin_round(x))


def gcd(a: int, b: int) -> int:
    """
    Greatest common divisor of two integers using Euclidean algorithm.

    Reference: PGauxiliaryFunctions.pl (via Euclidean algorithm)

    Example:
        gcd(20, 30) → 10
    """
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def gcf(*args: int) -> int:
    """
    Greatest common factor of multiple integers.

    Reference: PGauxiliaryFunctions.pl::gcf

    Example:
        gcf(20, 30, 45) → 5
    """
    if not args:
        raise ValueError("gcf requires at least one argument")

    # Filter out zeros and non-positive - but keep track for compatibility
    filtered = [abs(x) for x in args if x != 0]

    if not filtered:
        raise ValueError("Cannot take gcf of all-zero set")

    return _reduce(gcd, filtered)


# Alias gcd for PGauxiliaryFunctions.pl compatibility
def _gcd(*args: int) -> int:
    """Alias for gcf - greatest common divisor."""
    return gcf(*args)


def lcm(a: int, b: int) -> int:
    """
    Least common multiple of two integers.

    Reference: PGauxiliaryFunctions.pl::lcm

    Example:
        lcm(4, 6) → 12
    """
    a, b = abs(a), abs(b)
    if a == 0 or b == 0:
        return 0
    return (a * b) // gcd(a, b)


def lcm_multiple(*args: int) -> int:
    """
    Least common multiple of multiple integers.

    Reference: PGauxiliaryFunctions.pl::lcm

    Example:
        lcm_multiple(3, 4, 5, 6) → 60
    """
    if not args:
        raise ValueError("lcm requires at least one argument")

    # Filter out zeros
    filtered = [abs(x) for x in args if x != 0]
    if not filtered:
        return 0

    return _reduce(lcm, filtered)


def isPrime(n: int) -> bool:
    """
    Test if n is prime.

    Returns True if n is prime, False otherwise.

    Reference: PGauxiliaryFunctions.pl::isPrime

    Example:
        isPrime(7) → True
        isPrime(8) → False
    """
    n = int(n)
    if n == 2 or n == 3:
        return True
    if n == 1 or n == 0 or n < 0:
        return False
    if n % 2 == 0:
        return False

    # Check odd divisors up to sqrt(n)
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def reduce(numerator: int, denominator: int) -> Tuple[int, int]:
    """
    Reduce a fraction to lowest terms.

    Returns (reduced_numerator, reduced_denominator).

    Reference: PGauxiliaryFunctions.pl::reduce

    Example:
        reduce(15, 20) → (3, 4)
        reduce(-6, 9) → (-2, 3)
    """
    if denominator == 0:
        raise ValueError("denominator cannot be zero")

    g = gcf(abs(numerator), abs(denominator))
    num = numerator // g
    den = denominator // g

    # Ensure only numerator is negative
    if (num < 0) != (den < 0):
        num = -abs(num)
        den = abs(den)
    else:
        num = abs(num)
        den = abs(den)

    return (num, den)


def preformat(coefficient: Union[int, float], variable_str: str) -> str:
    """
    Format coefficient with variable for polynomial display.

    Handles special cases: 0, 1, -1.

    Reference: PGauxiliaryFunctions.pl::preformat

    Example:
        preformat(-1, "\\pi") → "-\\pi"
        preformat(1, "x") → "x"
        preformat(2, "x^2") → "2 x^2"
        preformat(0, "x") → "0"
    """
    if coefficient == 0:
        return "0"
    elif coefficient == 1:
        return variable_str if variable_str else "1"
    elif coefficient == -1:
        return f"-{variable_str}" if variable_str else "-1"
    else:
        return f"{coefficient} {variable_str}" if variable_str else str(coefficient)
def _expand_coprime_array(arr):
    """Flatten arrays or ranges passed to coprime helpers."""
    if isinstance(arr, range):
        return list(arr)
    if isinstance(arr, (list, tuple)):
        expanded = []
        for item in arr:
            if isinstance(item, range):
                expanded.extend(list(item))
            else:
                expanded.append(item)
        return expanded
    return [arr]



def random_coprime(
    *array_refs: List[int],
) -> Union[Tuple[int, ...], List[int]]:
    """
    Generate random n-tuple of coprime integers.

    Each integer comes from corresponding array. All integers in the tuple
    are coprime (gcd = 1).

    Reference: PGauxiliaryFunctions.pl::random_coprime

    Example:
        random_coprime([1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9])
        May return (2,9) or (1,1) but not (6,8)
    """
    if not array_refs:
        raise ValueError("random_coprime requires at least one array")

    # Convert single arrays or mixed iterables to list refs
    arrays = [_expand_coprime_array(arr) for arr in array_refs]

    # Start with first array as 1-tuples
    candidates = [[x] for x in arrays[0]]

    if not candidates:
        raise ValueError("Unable to find a coprime tuple from input")

    # Process remaining arrays
    for next_arr in arrays[1:]:
        new_candidates = []
        for candidate_tuple in candidates:
            for next_val in next_arr:
                # Calculate gcd of all values including next_val
                all_vals = [abs(x) for x in (candidate_tuple + [next_val]) if x != 0]

                if not all_vals:
                    # All zeros - not coprime
                    continue
                elif len(all_vals) == 1:
                    # Single non-zero value is "coprime"
                    new_candidates.append(candidate_tuple + [next_val])
                else:
                    # Check if gcd of all is 1
                    overall_gcd = _reduce(gcd, all_vals)
                    if overall_gcd == 1:
                        new_candidates.append(candidate_tuple + [next_val])

        if not new_candidates:
            raise ValueError("Unable to find a coprime tuple from input")

        candidates = new_candidates

    # Return random selection from valid candidates
    result = _random.choice(candidates)
    return tuple(result)


def random_pairwise_coprime(
    *array_refs: List[int],
) -> Union[Tuple[int, ...], List[int]]:
    """
    Generate random n-tuple where all pairs are coprime.

    Stronger constraint than random_coprime: gcd of any two elements = 1.

    Reference: PGauxiliaryFunctions.pl::random_pairwise_coprime

    Example:
        random_pairwise_coprime([-9..9], [1..9], [1..9])
        May return (-3,7,4) or (-1,1,1) but not (-2,2,3) or (3,5,6)
    """
    if not array_refs:
        raise ValueError("random_pairwise_coprime requires at least one array")

    # Convert to list refs
    arrays = [_expand_coprime_array(arr) for arr in array_refs]

    # Start with first array as 1-tuples
    candidates = [[x] for x in arrays[0]]

    if not candidates:
        raise ValueError("Unable to find a coprime tuple from input")

    # Process remaining arrays
    for next_arr in arrays[1:]:
        new_candidates = []
        for candidate_tuple in candidates:
            for next_val in next_arr:
                # Check if next_val is coprime with all values in tuple (pairwise)
                is_pairwise_coprime = True
                for candidate_val in candidate_tuple:
                    if next_val == 0 and candidate_val == 0:
                        is_pairwise_coprime = False
                        break
                    if gcd(abs(next_val), abs(candidate_val)) != 1:
                        is_pairwise_coprime = False
                        break

                if is_pairwise_coprime:
                    new_candidates.append(candidate_tuple + [next_val])

        if not new_candidates:
            raise ValueError("Unable to find a pairwise coprime tuple from input")

        candidates = new_candidates

    # Return random selection from valid candidates
    result = _random.choice(candidates)
    return tuple(result)


def factorial(n: int) -> int:
    """
    Compute factorial of n.

    Reference: PGauxiliaryFunctions.pl (factorial function)

    Example:
        factorial(5) → 120
    """
    return math.factorial(n)
