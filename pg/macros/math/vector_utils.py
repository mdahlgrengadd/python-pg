"""Vector Utilities for WeBWorK.

This module provides vector manipulation and geometric utilities including
norm, unit vector calculations, and geometric line representation.

Reference: macros/graph/parserVectorUtils.pl
Based on Value.pm from the Perl WeBWorK distribution.
"""

import math
import random
from typing import List, Optional, Tuple, Union

def _random_step_value(low: float, high: float, step: float | None = None):
    """Return a random value between low and high using the provided step."""
    if step in (None, 0):
        return random.randint(int(low), int(high))

    if step < 0:
        raise ValueError("step must be positive")

    steps = int(round((high - low) / step))
    if steps < 0:
        raise ValueError("Invalid range for random selection")

    index = random.randint(0, steps)
    return low + index * step


def norm(vector: Union[List[float], Tuple[float, ...], 'Vector']) -> float:
    """
    Compute the norm (magnitude/length) of a vector.
    
    The norm is calculated as: ||v|| = âˆš(vâ‚Â² + vâ‚‚Â² + ... + vâ‚™Â²)
    
    Args:
        vector: A vector as a list, tuple, or Vector object
        
    Returns:
        The magnitude of the vector
        
    Example:
        >>> norm([3, 4])
        5.0
        >>> norm((1, 1, 1))
        1.7320508075688772
    
    Perl Source: Value.pm Vector methods
    """
    if hasattr(vector, 'norm'):
        # If it's a Vector object with norm method
        return vector.norm()
    
    # Fallback for list/tuple
    if not vector:
        return 0.0
    
    sum_of_squares = sum(x**2 for x in vector)
    return math.sqrt(sum_of_squares)


def unit(vector: Union[List[float], Tuple[float, ...], 'Vector']) -> Union[List[float], 'Vector']:
    """
    Compute the unit vector in the direction of the given vector.
    
    The unit vector is: u = v / ||v||
    
    Args:
        vector: A vector as a list, tuple, or Vector object
        
    Returns:
        Unit vector in the same direction as input
        
    Raises:
        ValueError: If the vector is zero (magnitude is 0)
        
    Example:
        >>> unit([3, 4])
        [0.6, 0.8]
        >>> unit((1, 0, 0))
        [1.0, 0.0, 0.0]
    
    Perl Source: Value.pm Vector methods
    """
    if hasattr(vector, 'unit'):
        # If it's a Vector object with unit method
        return vector.unit()
    
    # Fallback for list/tuple
    magnitude = norm(vector)
    
    if magnitude == 0:
        raise ValueError("Cannot compute unit vector of zero vector")
    
    # Return as list (same type as input)
    return [x / magnitude for x in vector]


class Line:
    """
    Geometric line representation in 2D or 3D space.
    
    A line can be defined by a point and a direction vector, or by
    two points that the line passes through.
    
    Attributes:
        point: A point (base point) on the line
        direction: Direction vector of the line
        point2: Optional second point for parametric definition
    """
    
    def __init__(self, point: Union[List, Tuple] = None, direction: Union[List, Tuple] = None, 
                 point2: Union[List, Tuple] = None, **kwargs):
        """
        Create a geometric line.
        
        Args:
            point: Base point on the line (e.g., [1, 2] in 2D)
            direction: Direction vector of the line (e.g., [3, 4])
            point2: Alternative second point to define the line through two points
            **kwargs: Additional options
            
        Example:
            >>> line = Line([0, 0], [1, 1])  # Line through origin with direction (1,1)
            >>> line2 = Line([1, 2], point2=[3, 4])  # Line through two points
        
        Perl Source: Value.pm Line class
        """
        self.point = point or []
        self.direction = direction or []
        self.point2 = point2
        self.options = kwargs
        
        # If point2 is provided, calculate direction vector
        if self.point2 and self.point:
            self.direction = [self.point2[i] - self.point[i] for i in range(len(self.point))]
    
    def evaluate(self, t: float) -> List[float]:
        """
        Evaluate the line at parameter value t.
        
        Returns: point + t * direction
        
        Args:
            t: Parameter value
            
        Returns:
            Point on the line at parameter t
            
        Example:
            >>> line = Line([0, 0], [1, 1])
            >>> line.evaluate(2)
            [2, 2]
        """
        return [self.point[i] + t * self.direction[i] for i in range(len(self.point))]
    
    def __str__(self) -> str:
        """Return string representation."""
        if self.point and self.direction:
            return f"Line through {self.point} with direction {self.direction}"
        return "Line"
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"Line({self.point}, {self.direction})"


# Formatting constants and functions

def Overline(name: str) -> str:
    r"""
    Format a vector name with overline notation.

    Returns: "\overline{" + name + "}" for LaTeX display

    Reference: parserVectorUtils.pl::Overline

    Example:
        >>> Overline("v")
        "\\overline{v}"
    """
    return f"\\overline{{{name}}}"


def BoldMath(name: str) -> str:
    """
    Format a vector name in bold math notation.

    Returns: "\\mathbf{" + name + "}"

    Reference: parserVectorUtils.pl::BoldMath

    Example:
        >>> BoldMath("v")
        "\\mathbf{v}"
    """
    return f"\\mathbf{{{name}}}"


# Gradient/Nabla symbol
GRAD = "\\nabla"


# Random vector/point generators

def non_zero_vector(
    low: float = -9,
    high: float = 9,
    dimension: int = 2,
    step: float | None = 1,
) -> List[float]:
    """
    Generate a random non-zero vector with components spaced by `step`.
    """
    while True:
        vector = [_random_step_value(low, high, step) for _ in range(dimension)]
        if any(v != 0 for v in vector):
            return vector


def non_zero_vector2D(low: float = -9, high: float = 9, step: float | None = 1) -> List[float]:
    """Generate a random non-zero 2D vector."""
    return non_zero_vector(low, high, 2, step)



def non_zero_vector3D(low: float = -9, high: float = 9, step: float | None = 1):
    """
    Generate a random non-zero 3D vector as a Vector MathObject.

    Returns a Vector object with 3 components, all non-zero.
    """
    from pg.math.geometric import Vector
    components = non_zero_vector(low, high, 3, step)
    return Vector(components)



def non_zero_point(
    low: float = -9,
    high: float = 9,
    dimension: int = 2,
    step: float | None = 1,
) -> List[float]:
    """Generate a random non-zero point."""
    return non_zero_vector(low, high, dimension, step)


def non_zero_point2D(low: float = -9, high: float = 9, step: float | None = 1) -> List[float]:
    """Generate a random non-zero 2D point."""
    return non_zero_point(low, high, 2, step)



def non_zero_point3D(low: float = -9, high: float = 9, step: float | None = 1) -> List[float]:
    """Generate a random non-zero 3D point."""
    return non_zero_point(low, high, 3, step)


class Plane:
    """
    Representation of a plane in 3D space.

    A plane can be defined by a normal vector and a point, or by coefficients
    in the equation ax + by + cz = d.

    Reference: parserVectorUtils.pl

    Attributes:
        normal: Normal vector to the plane
        point: A point on the plane
        d: Constant term (ax + by + cz = d)
    """

    def __init__(
        self,
        normal: Union[List[float], Tuple[float, ...], None] = None,
        point: Union[List[float], Tuple[float, ...], None] = None,
        d: Optional[float] = None,
    ):
        """
        Create a plane.

        Args:
            normal: Normal vector [a, b, c] for equation ax + by + cz = d
            point: A point on the plane
            d: Constant d in equation (calculated if not provided with point)

        Example:
            >>> plane = Plane([1, 0, 0], [1, 0, 0])  # x = 1
            >>> plane = Plane([1, 1, 1])  # x + y + z = d
        """
        self.normal = list(normal) if normal else [0, 0, 1]
        self.point = list(point) if point else [0, 0, 0]

        # Calculate d from plane equation: normal Â· (point - origin) = d
        if d is None and point:
            self.d = sum(self.normal[i] * self.point[i] for i in range(3))
        else:
            self.d = d or 0

    def contains_point(self, point: Union[List[float], Tuple[float, ...]]) -> bool:
        """
        Check if a point lies on the plane.

        Args:
            point: Point to check

        Returns:
            True if point is on the plane (within numerical tolerance)
        """
        val = sum(self.normal[i] * point[i] for i in range(3))
        return abs(val - self.d) < 1e-10

    def __str__(self) -> str:
        """Return string representation."""
        a, b, c = self.normal
        return f"{a}x + {b}y + {c}z = {self.d}"

    def __repr__(self) -> str:
        """Return developer representation."""
        return f"Plane({self.normal}, {self.point})"


__all__ = [
    "norm",
    "unit",
    "Line",
    "Overline",
    "BoldMath",
    "GRAD",
    "non_zero_vector",
    "non_zero_vector2D",
    "non_zero_vector3D",
    "non_zero_point",
    "non_zero_point2D",
    "non_zero_point3D",
    "Plane",
]

