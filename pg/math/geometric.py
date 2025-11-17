"""
Geometric MathValue types: Point, Vector, Matrix.

These types represent geometric objects with specialized operations.

Reference: lib/Value/Point.pm, lib/Value/Vector.pm, lib/Value/Matrix.pm
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .numeric import Real
from .value import MathValue, ToleranceMode, TypePrecedence


class Point(MathValue):
    """
    Point in n-dimensional space.

    Represented as ordered coordinates (x, y, z, ...).
    Points support distance calculations but not vector operations.

    Reference: lib/Value/Point.pm
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    coordinates: list[MathValue] = Field(default_factory=list)
    context: Any | None = None
    type_precedence: TypePrecedence = Field(default=TypePrecedence.POINT, init=False)

    def __init__(
        self,
        *args: Any,
        context: Any | None = None,
        coordinates: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a Point supporting legacy call patterns."""
        processed_context = context
        if coordinates is not None and args:
            raise ValueError("Point accepts either coordinates or positional arguments, not both")

        if coordinates is not None:
            processed_coords = self._coerce_coordinate_values(coordinates)
        else:
            processed_coords, processed_context = self._parse_arguments(args, processed_context)

        super().__init__(coordinates=processed_coords, context=processed_context, **kwargs)

    @staticmethod
    def _coerce_coordinate_values(raw_coords: Iterable[Any]) -> list[MathValue]:
        """Convert Python/native coordinates into MathValue instances."""
        from .value import MathValue as MV

        return [coord if isinstance(coord, MathValue) else MV.from_python(coord) for coord in raw_coords]

    @classmethod
    def _parse_arguments(
        cls,
        args: tuple[Any, ...],
        context: Any | None,
    ) -> tuple[list[MathValue], Any | None]:
        """Parse positional constructor arguments."""
        if len(args) == 0:
            return cls._coerce_coordinate_values([]), context

        if len(args) == 1:
            single = args[0]
            if isinstance(single, str):
                coords, resolved_context = cls._parse_string_literal(single, context)
                return cls._coerce_coordinate_values(coords), resolved_context
            if isinstance(single, (list, tuple)):
                return cls._coerce_coordinate_values(list(single)), context

        return cls._coerce_coordinate_values(list(args)), context

    @classmethod
    def _parse_string_literal(
        cls,
        literal: str,
        context: Any | None,
    ) -> tuple[list[Any], Any | None]:
        """Parse string literal input using Compute to match Perl behavior."""
        if context is None:
            from .context import get_current_context

            context = get_current_context()

        from .compute import Compute

        parsed = Compute(literal, context)
        coords = cls._extract_coordinates(parsed)
        return coords, context

    @staticmethod
    def _extract_coordinates(parsed: Any) -> list[Any]:
        """Extract coordinate sequence from parsed objects."""
        if isinstance(parsed, Point):
            return list(parsed.coords)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        if hasattr(parsed, 'coords'):
            return list(parsed.coords)
        if hasattr(parsed, 'to_python'):
            py_val = parsed.to_python()
            if isinstance(py_val, (list, tuple)):
                return list(py_val)
            if isinstance(py_val, str):
                import re

                match = re.match(r'\(([^)]+)\)', py_val.strip())
                if match:
                    coords_str = match.group(1)
                    return [c.strip() for c in coords_str.split(',')]
                return [parsed]
            return [py_val]
        return [parsed]

    @property
    def coords(self) -> list[MathValue]:
        """Backward-compatible coords accessor."""
        return self.coordinates

    @coords.setter
    def coords(self, value: Iterable[Any]) -> None:
        self.coordinates = self._coerce_coordinate_values(list(value))

    @field_validator("coordinates", mode="before")
    @classmethod
    def _validate_coordinates(cls, value):
        if value is None:
            return []
        return cls._coerce_coordinate_values(value if isinstance(value, (list, tuple)) else list(value))

    def promote(self, other: MathValue) -> MathValue:
        """Points don't promote to other types."""
        return self

    def compare(
        self, other: MathValue, tolerance: float = 0.001, mode: str = ToleranceMode.RELATIVE
    ) -> bool:
        """Compare points coordinate-wise."""
        if not isinstance(other, Point):
            return False

        if len(self.coords) != len(other.coords):
            return False

        return all(c1.compare(c2, tolerance, mode) for c1, c2 in zip(self.coords, other.coords))

    def to_string(self) -> str:
        """Convert to string."""
        coords_str = ", ".join(c.to_string() for c in self.coords)
        return f"({coords_str})"

    def to_tex(self) -> str:
        """Convert to LaTeX."""
        coords_str = ", ".join(c.to_tex() for c in self.coords)
        return f"\\left({coords_str}\\right)"

    def to_python(self) -> tuple[float, ...]:
        """Convert to Python tuple."""
        return tuple(c.to_python() for c in self.coords)

    def __len__(self) -> int:
        """Dimension of the point."""
        return len(self.coords)

    def __getitem__(self, index: int) -> MathValue:
        """Get coordinate by index."""
        return self.coords[index]

    def distance(self, other: Point) -> Real:
        """
        Calculate Euclidean distance to another point.

        Args:
            other: Another point

        Returns:
            Real number representing the distance
        """
        if len(self.coords) != len(other.coords):
            raise ValueError("Points must have same dimension")

        sum_sq = sum((c1.to_python() - c2.to_python()) **
                     2 for c1, c2 in zip(self.coords, other.coords))
        return Real(math.sqrt(sum_sq))

    # Arithmetic operators (limited for points)

    def __add__(self, other: Any) -> Point:
        """Vector addition (point + vector = point)."""
        if isinstance(other, (Point, Vector)):
            if len(self.coords) != len(other.coords if hasattr(other, 'coords') else other.components):
                raise ValueError("Dimensions must match")
            other_coords = other.coords if isinstance(
                other, Point) else other.components
            return Point([c1 + c2 for c1, c2 in zip(self.coords, other_coords)])
        else:
            return NotImplemented

    def __radd__(self, other: Any) -> Point:
        """Right addition."""
        return self.__add__(other)

    def __sub__(self, other: Any) -> MathValue:
        """Subtraction: point - point = vector, point - vector = point."""
        if isinstance(other, Point):
            # point - point = vector
            if len(self.coords) != len(other.coords):
                raise ValueError("Dimensions must match")
            return Vector([c1 - c2 for c1, c2 in zip(self.coords, other.coords)])
        elif isinstance(other, Vector):
            # point - vector = point
            if len(self.coords) != len(other.components):
                raise ValueError("Dimensions must match")
            return Point([c1 - c2 for c1, c2 in zip(self.coords, other.components)])
        else:
            return NotImplemented

    def __rsub__(self, other: Any) -> MathValue:
        """Right subtraction."""
        if isinstance(other, Point):
            return Point(other.coords) - self
        else:
            return NotImplemented

    # Not supported for points

    def __mul__(self, other: Any) -> MathValue:
        """Multiplication not supported."""
        raise TypeError("Point does not support multiplication")

    def __rmul__(self, other: Any) -> MathValue:
        """Right multiplication not supported."""
        raise TypeError("Point does not support multiplication")

    def __truediv__(self, other: Any) -> Point:
        """
        Scalar division: point / scalar = point.

        Used for computing midpoints: (p1 + p2) / 2
        """
        from .numeric import Real

        # Convert to scalar if needed
        if isinstance(other, (int, float)):
            scalar = other
        elif isinstance(other, Real):
            scalar = other.to_python()
        else:
            raise TypeError(f"Cannot divide Point by {type(other).__name__}")

        if scalar == 0:
            raise ZeroDivisionError("Cannot divide point by zero")

        return Point([c / scalar for c in self.coords])

    def __rtruediv__(self, other: Any) -> MathValue:
        """Right division not supported (scalar / point makes no sense)."""
        raise TypeError("Cannot divide scalar by Point")

    def __pow__(self, other: Any) -> MathValue:
        """Power not supported."""
        raise TypeError("Point does not support exponentiation")

    def __rpow__(self, other: Any) -> MathValue:
        """Right power not supported."""
        raise TypeError("Point does not support exponentiation")

    def __neg__(self) -> Point:
        """Negation."""
        return Point([-c for c in self.coords])

    def __pos__(self) -> Point:
        """Unary positive."""
        return Point([+c for c in self.coords])

    def __abs__(self) -> Real:
        """Distance from origin (magnitude)."""
        sum_sq = sum(c.to_python() ** 2 for c in self.coords)
        return Real(math.sqrt(sum_sq))


class Vector(MathValue):
    """
    Vector in n-dimensional space.

    Supports vector operations: dot product, cross product, norm, etc.

    Reference: lib/Value/Vector.pm
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    components: list[MathValue] = Field(default_factory=list)
    context: Any | None = None
    type_precedence: TypePrecedence = Field(default=TypePrecedence.VECTOR, init=False)

    def __init__(
        self,
        *args: Any,
        context: Any | None = None,
        components: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a Vector supporting legacy construction patterns."""
        if components is not None and args:
            raise ValueError("Vector accepts either components or positional arguments, not both")

        if components is not None:
            parsed_components = self._coerce_components(components)
            resolved_context = context
        else:
            parsed_components, resolved_context = self._parse_arguments(args, context)

        super().__init__(components=parsed_components, context=resolved_context, **kwargs)

    @staticmethod
    def _coerce_components(raw_components: Iterable[Any]) -> list[MathValue]:
        """Convert raw component values into MathValue instances."""
        from .value import MathValue as MV

        return [comp if isinstance(comp, MathValue) else MV.from_python(comp) for comp in raw_components]

    @classmethod
    def _parse_arguments(
        cls,
        args: tuple[Any, ...],
        context: Any | None,
    ) -> tuple[list[MathValue], Any | None]:
        """Parse constructor arguments into vector components."""
        if len(args) == 0:
            return cls._coerce_components([]), context

        if len(args) == 1:
            single = args[0]
            if isinstance(single, str):
                literal_components = cls._parse_vector_literal(single, context)
                return cls._coerce_components(literal_components), context
            if isinstance(single, (list, tuple)):
                return cls._coerce_components(list(single)), context
            if isinstance(single, MathValue) and hasattr(single, 'to_string'):
                literal_components = cls._parse_vector_literal(single.to_string(), context)
                return cls._coerce_components(literal_components), context

        return cls._coerce_components(list(args)), context

    @field_validator("components", mode="before")
    @classmethod
    def _validate_components(cls, value):
        if value is None:
            return []
        return cls._coerce_components(value if isinstance(value, (list, tuple)) else list(value))

    def promote(self, other: MathValue) -> MathValue:
        """Vectors don't promote to other types."""
        return self

    def compare(
        self, other: MathValue, tolerance: float = 0.001, mode: str = ToleranceMode.RELATIVE
    ) -> bool:
        """Compare vectors component-wise."""
        if not isinstance(other, Vector):
            return False

        if len(self.components) != len(other.components):
            return False

        return all(
            c1.compare(c2, tolerance, mode) for c1, c2 in zip(self.components, other.components)
        )

    def to_string(self) -> str:
        """Convert to string."""
        comps_str = ", ".join(c.to_string() for c in self.components)
        return f"<{comps_str}>"

    def to_tex(self) -> str:
        """Convert to LaTeX."""
        comps_str = ", ".join(c.to_tex() for c in self.components)
        return f"\\left\\langle {comps_str} \\right\\rangle"

    def to_python(self) -> list[float]:
        """Convert to Python list."""
        return [c.to_python() for c in self.components]

    def to_numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
        return np.array(self.to_python())

    @property
    def value(self) -> list[float]:
        """
        Get the numeric value of the vector as a list.

        Perl compatibility property - returns the components as Python values.

        Returns:
            List of float values
        """
        return self.to_python()

    def cmp(self, *args, **kwargs) -> "MathValue":
        """
        Return a comparator for this vector.

        In Perl MathObjects, cmp() returns a comparator object.
        For Python, we return self to maintain compatibility.
        """
        return self

    def __len__(self) -> int:
        """Dimension of the vector."""
        return len(self.components)

    def __getitem__(self, index: int) -> MathValue:
        """Get component by index."""
        return self.components[index]

    # Vector operations

    def norm(self) -> Real:
        """
        Calculate the Euclidean norm (magnitude) of the vector.

        Returns:
            Real number representing ||v||
        """
        sum_sq = sum(c.to_python() ** 2 for c in self.components)
        return Real(math.sqrt(sum_sq))

    def unit(self) -> Vector:
        """
        Return the unit vector (normalized).

        Returns:
            Vector with norm 1
        """
        magnitude = self.norm().value
        if magnitude == 0:
            raise ValueError("Cannot normalize zero vector")
        return Vector([c / magnitude for c in self.components])

    def dot(self, other: Vector) -> Real:
        """
        Dot product with another vector.

        Args:
            other: Another vector

        Returns:
            Real number (scalar)
        """
        if len(self.components) != len(other.components):
            raise ValueError("Vectors must have same dimension")

        result = sum(
            c1.to_python() * c2.to_python()
            for c1, c2 in zip(self.components, other.components)
        )
        return Real(result)

    def cross(self, other: Vector) -> Vector:
        """
        Cross product with another vector (3D only).

        Args:
            other: Another 3D vector

        Returns:
            Vector perpendicular to both
        """
        if len(self.components) != 3 or len(other.components) != 3:
            raise ValueError("Cross product only defined for 3D vectors")

        a1, a2, a3 = [c.to_python() for c in self.components]
        b1, b2, b3 = [c.to_python() for c in other.components]

        return Vector([a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1])

    def is_parallel(self, other: Vector, tolerance: float = 0.001) -> bool:
        """Check if vectors are parallel."""
        # Vectors are parallel if cross product is zero (3D)
        # Or if one is a scalar multiple of the other
        if len(self.components) == 3 and len(other.components) == 3:
            cross = self.cross(other)
            return cross.norm().value < tolerance
        else:
            # Check if ratios are constant
            ratios = []
            for c1, c2 in zip(self.components, other.components):
                v1, v2 = c1.to_python(), c2.to_python()
                if abs(v2) > tolerance:
                    ratios.append(v1 / v2)
                elif abs(v1) > tolerance:
                    return False  # One is zero but other isn't
            return all(abs(r - ratios[0]) < tolerance for r in ratios) if ratios else True

    def is_orthogonal(self, other: Vector, tolerance: float = 0.001) -> bool:
        """Check if vectors are orthogonal (perpendicular)."""
        return abs(self.dot(other).value) < tolerance

    def extract(self, index: int) -> MathValue:
        """
        Extract a single component of the vector by 1-based index.

        Perl compatibility method - uses 1-based indexing.

        Args:
            index: Component index (1-based, so 1 is first component)

        Returns:
            The component at that position

        Raises:
            IndexError: If index is out of range
        """
        # Convert from 1-based (Perl) to 0-based (Python) indexing
        py_index = index - 1
        if py_index < 0 or py_index >= len(self.components):
            raise IndexError(
                f"Vector component {index} out of range (vector has {len(self.components)} components)")
        return self.components[py_index]

    def __getattr__(self, name: str) -> MathValue:
        """
        Support component access via v0, v1, v2, etc. attributes.

        Also supports isParallel, isOrthogonal as aliases.

        Args:
            name: Attribute name

        Returns:
            Component value for v0/v1/v2, or method for is* methods
        """
        # Handle component access: v0, v1, v2, etc. (0-based indexing)
        if name.startswith("v") and len(name) > 1 and name[1:].isdigit():
            index = int(name[1:])
            if index < 0 or index >= len(self.components):
                raise AttributeError(
                    f"'Vector' object has no attribute '{name}' "
                    f"(vector has {len(self.components)} components, use v0-v{len(self.components)-1})"
                )
            return self.components[index]

        # Handle method aliases
        if name == "isParallel":
            return self.is_parallel
        if name == "isOrthogonal":
            return self.is_orthogonal

        raise AttributeError(f"'Vector' object has no attribute '{name}'")

    @staticmethod
    @staticmethod
    def _parse_vector_literal(expr: str, context: Any | None) -> list[Any]:
        expr = expr.strip()
        if not (expr.startswith('<') and expr.endswith('>')):
            return [expr]

        content = expr[1:-1]
        components: list[str] = []
        current = []
        depth = 0
        for ch in content:
            if ch in '([{':
                depth += 1
            elif ch in ')]}':
                depth -= 1
            if ch == ',' and depth == 0:
                components.append(''.join(current).strip())
                current = []
                continue
            current.append(ch)
        if current:
            components.append(''.join(current).strip())

        from pg.math.context import get_current_context
        from pg.math.compute import Compute

        ctx = context or get_current_context()
        parsed: list[Any] = []
        for comp in components:
            if not comp:
                continue
            parsed.append(Compute(comp, context=ctx))

        return parsed

    def answer_checker(self, **options):
        """
        Create an answer checker for this Vector.

        Args:
            **options: Checker options (tolerance, tolType, checker)

        Returns:
            VectorAnswerChecker that can check student answers
        """
        from .answer_checker import VectorAnswerChecker
        return VectorAnswerChecker(self, **options)

    def cmp(self, **options):
        """
        Alias for answer_checker() - Perl compatibility.

        Args:
            **options: Checker options (tolerance, tolType, checker)

        Returns:
            VectorAnswerChecker that can check student answers
        """
        return self.answer_checker(**options)

    # Arithmetic operators

    def __add__(self, other: Any) -> Vector:
        """Vector addition."""
        if isinstance(other, Vector):
            if len(self.components) != len(other.components):
                raise ValueError("Vectors must have same dimension")
            return Vector([c1 + c2 for c1, c2 in zip(self.components, other.components)])
        else:
            return NotImplemented

    def __radd__(self, other: Any) -> Vector:
        """Right addition."""
        return self.__add__(other)

    def __sub__(self, other: Any) -> Vector:
        """Vector subtraction."""
        if isinstance(other, Vector):
            if len(self.components) != len(other.components):
                raise ValueError("Vectors must have same dimension")
            return Vector([c1 - c2 for c1, c2 in zip(self.components, other.components)])
        else:
            return NotImplemented

    def __rsub__(self, other: Any) -> Vector:
        """Right subtraction."""
        if isinstance(other, Vector):
            return Vector(other.components) - self
        else:
            return NotImplemented

    def __mul__(self, other: Any) -> MathValue:
        """Scalar multiplication or dot product."""
        if isinstance(other, (int, float, Real)):
            # Scalar multiplication
            scalar = other.value if isinstance(other, Real) else other
            return Vector([c * scalar for c in self.components])
        elif isinstance(other, Vector):
            # Dot product
            return self.dot(other)
        else:
            return NotImplemented

    def __rmul__(self, other: Any) -> Vector:
        """Right scalar multiplication."""
        if isinstance(other, (int, float, Real)):
            scalar = other.value if isinstance(other, Real) else other
            return Vector([scalar * c for c in self.components])
        else:
            return NotImplemented

    def __truediv__(self, other: Any) -> Vector:
        """Scalar division."""
        if isinstance(other, (int, float, Real)):
            scalar = other.value if isinstance(other, Real) else other
            if scalar == 0:
                raise ZeroDivisionError("Vector division by zero")
            return Vector([c / scalar for c in self.components])
        else:
            return NotImplemented

    def __rtruediv__(self, other: Any) -> MathValue:
        """Right division not supported."""
        raise TypeError("Cannot divide scalar by vector")

    def __pow__(self, other: Any) -> MathValue:
        """Power not supported."""
        raise TypeError("Vector does not support exponentiation")

    def __rpow__(self, other: Any) -> MathValue:
        """Right power not supported."""
        raise TypeError("Vector does not support exponentiation")

    def __neg__(self) -> Vector:
        """Negation."""
        return Vector([-c for c in self.components])

    def __pos__(self) -> Vector:
        """Unary positive."""
        return Vector([+c for c in self.components])

    def __abs__(self) -> Real:
        """Magnitude (norm)."""
        return self.norm()


class Matrix(MathValue):
    """
    Matrix (2D array) with matrix operations.

    Supports matrix multiplication, transpose, determinant, inverse, etc.

    Reference: lib/Value/Matrix.pm
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    rows: list[list[MathValue]] = Field(default_factory=list)
    context: Any | None = None
    type_precedence: TypePrecedence = Field(default=TypePrecedence.MATRIX, init=False)

    def __init__(
        self,
        rows: Iterable[Iterable[Any]] | np.ndarray,
        context: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a Matrix ensuring rectangular structure."""
        processed_rows = self._coerce_rows(rows)
        super().__init__(rows=processed_rows, context=context, **kwargs)

    @staticmethod
    def _coerce_rows(raw_rows: Any) -> list[list[MathValue]]:
        """Convert raw row iterables into MathValue rows."""
        from .value import MathValue as MV

        if isinstance(raw_rows, Matrix):
            return [[cell for cell in row] for row in raw_rows.rows]

        if isinstance(raw_rows, np.ndarray):
            raw_rows = raw_rows.tolist()

        if not isinstance(raw_rows, Iterable):
            raise TypeError("Matrix rows must be iterable sequences")

        normalized: list[list[MathValue]] = []
        for row in raw_rows:
            if isinstance(row, np.ndarray):
                row = row.tolist()
            if not isinstance(row, Iterable):
                raise TypeError("Matrix rows must be iterable sequences")
            normalized.append([cell if isinstance(cell, MathValue) else MV.from_python(cell) for cell in row])

        if normalized:
            row_len = len(normalized[0])
            if not all(len(row) == row_len for row in normalized):
                raise ValueError("Matrix rows must all have same length")
        return normalized

    @field_validator("rows", mode="before")
    @classmethod
    def _validate_rows(cls, value):
        if value is None:
            return []
        return cls._coerce_rows(value)

    def promote(self, other: MathValue) -> MathValue:
        """Matrices don't promote."""
        return self

    def compare(
        self, other: MathValue, tolerance: float = 0.001, mode: str = ToleranceMode.RELATIVE
    ) -> bool:
        """Compare matrices element-wise."""
        if not isinstance(other, Matrix):
            return False

        if self.shape != other.shape:
            return False

        for row1, row2 in zip(self.rows, other.rows):
            for el1, el2 in zip(row1, row2):
                if not el1.compare(el2, tolerance, mode):
                    return False
        return True

    def to_string(self) -> str:
        """Convert to string."""
        rows_str = ", ".join(
            "[" + ", ".join(el.to_string() for el in row) + "]" for row in self.rows
        )
        return f"[{rows_str}]"

    def to_tex(self) -> str:
        """Convert to LaTeX (pmatrix)."""
        rows_tex = " \\\\ ".join(
            " & ".join(el.to_tex() for el in row) for row in self.rows
        )
        return f"\\begin{{pmatrix}} {rows_tex} \\end{{pmatrix}}"

    def to_python(self) -> list[list[float]]:
        """Convert to Python nested list."""
        return [[el.to_python() for el in row] for row in self.rows]

    def to_numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
        return np.array(self.to_python())

    @property
    def shape(self) -> tuple[int, int]:
        """Get matrix dimensions (rows, cols)."""
        if len(self.rows) == 0:
            return (0, 0)
        return (len(self.rows), len(self.rows[0]))

    def cmp(self, *args, **kwargs) -> "MathValue":
        """
        Return a comparator for this matrix.

        In Perl MathObjects, cmp() returns a comparator object.
        For Python, we return self to maintain compatibility.
        """
        return self

    def __getitem__(self, index: tuple[int, int] | int) -> MathValue:
        """Get element by (row, col) or row."""
        if isinstance(index, tuple):
            row, col = index
            return self.rows[row][col]
        else:
            # Return entire row as list
            return self.rows[index]

    # Matrix operations

    @property
    def transpose(self) -> Matrix:
        """
        Return the transpose of the matrix.

        In Perl this is called as ->transpose (no parens), so we make it a property.
        """
        if len(self.rows) == 0:
            return Matrix([])

        n_cols = len(self.rows[0])
        transposed = [[self.rows[i][j]
                       for i in range(len(self.rows))] for j in range(n_cols)]
        return Matrix(transposed)

    def column(self, index: int) -> Matrix:
        """
        Extract a column from the matrix as a column vector.

        Args:
            index: Column index (1-based, following Perl convention)

        Returns:
            Column vector as a Matrix
        """
        # Convert from 1-based to 0-based indexing
        col_idx = index - 1
        if col_idx < 0 or col_idx >= len(self.rows[0]) if self.rows else True:
            raise IndexError(f"Column index {index} out of range")

        # Extract column as a list of single-element rows
        column_vector = [[self.rows[i][col_idx]]
                         for i in range(len(self.rows))]
        return Matrix(column_vector)

    def row(self, index: int) -> Matrix:
        """
        Extract a row from the matrix as a row vector.

        Args:
            index: Row index (1-based, following Perl convention)

        Returns:
            Row vector as a Matrix
        """
        # Convert from 1-based to 0-based indexing
        row_idx = index - 1
        if row_idx < 0 or row_idx >= len(self.rows):
            raise IndexError(f"Row index {index} out of range")

        # Return the row as a single-row matrix
        return Matrix([self.rows[row_idx]])

    def copy(self) -> Matrix:
        """
        Create a deep copy of the matrix.

        Returns:
            New Matrix with copied data
        """
        import copy
        return Matrix(copy.deepcopy(self.rows))

    def determinant(self) -> Real:
        """
        Calculate the determinant (square matrices only).

        Returns:
            Real number representing the determinant

        Raises:
            ValueError: If matrix is not square
        """
        rows, cols = self.shape
        if rows != cols:
            raise ValueError("Determinant only defined for square matrices")

        # Use NumPy for efficiency
        det = np.linalg.det(self.to_numpy())
        return Real(det)

    def inverse(self) -> Matrix:
        """
        Calculate the matrix inverse (square, non-singular matrices only).

        Returns:
            Inverse matrix

        Raises:
            ValueError: If matrix is singular or not square
        """
        rows, cols = self.shape
        if rows != cols:
            raise ValueError("Inverse only defined for square matrices")

        # Use NumPy for efficiency
        try:
            inv = np.linalg.inv(self.to_numpy())
            return Matrix(inv.tolist())
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is singular (not invertible)")

    def trace(self) -> Real:
        """Calculate the trace (sum of diagonal elements)."""
        rows, cols = self.shape
        if rows != cols:
            raise ValueError("Trace only defined for square matrices")

        trace_sum = sum(self.rows[i][i].to_python() for i in range(rows))
        return Real(trace_sum)

    # Arithmetic operators

    def __add__(self, other: Any) -> Matrix:
        """Matrix addition."""
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrices must have same dimensions")
            result = [
                [el1 + el2 for el1, el2 in zip(row1, row2)]
                for row1, row2 in zip(self.rows, other.rows)
            ]
            return Matrix(result)
        else:
            return NotImplemented

    def __radd__(self, other: Any) -> Matrix:
        """Right addition."""
        return self.__add__(other)

    def __sub__(self, other: Any) -> Matrix:
        """Matrix subtraction."""
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrices must have same dimensions")
            result = [
                [el1 - el2 for el1, el2 in zip(row1, row2)]
                for row1, row2 in zip(self.rows, other.rows)
            ]
            return Matrix(result)
        else:
            return NotImplemented

    def __rsub__(self, other: Any) -> Matrix:
        """Right subtraction."""
        if isinstance(other, Matrix):
            return Matrix(other.rows) - self
        else:
            return NotImplemented

    def __mul__(self, other: Any) -> MathValue:
        """Matrix multiplication or scalar multiplication."""
        if isinstance(other, (int, float, Real)):
            # Scalar multiplication
            scalar = other.value if isinstance(other, Real) else other
            result = [[el * scalar for el in row] for row in self.rows]
            return Matrix(result)
        elif isinstance(other, Matrix):
            # Matrix multiplication
            if self.shape[1] != other.shape[0]:
                raise ValueError(
                    f"Cannot multiply {self.shape} by {other.shape} matrices")

            # Use NumPy for efficiency
            result = np.matmul(self.to_numpy(), other.to_numpy())
            return Matrix(result.tolist())
        elif isinstance(other, Vector):
            # Matrix * Vector = Vector
            if self.shape[1] != len(other.components):
                raise ValueError(
                    f"Cannot multiply {self.shape} matrix by {len(other)} vector")

            result = np.matmul(self.to_numpy(), other.to_numpy())
            return Vector(result.tolist())
        else:
            return NotImplemented

    def __rmul__(self, other: Any) -> Matrix:
        """Right multiplication (scalar only)."""
        if isinstance(other, (int, float, Real)):
            scalar = other.value if isinstance(other, Real) else other
            result = [[scalar * el for el in row] for row in self.rows]
            return Matrix(result)
        else:
            return NotImplemented

    def __truediv__(self, other: Any) -> Matrix:
        """Scalar division."""
        if isinstance(other, (int, float, Real)):
            scalar = other.value if isinstance(other, Real) else other
            if scalar == 0:
                raise ZeroDivisionError("Matrix division by zero")
            result = [[el / scalar for el in row] for row in self.rows]
            return Matrix(result)
        else:
            return NotImplemented

    def __rtruediv__(self, other: Any) -> MathValue:
        """Right division not supported."""
        raise TypeError("Cannot divide scalar by matrix")

    def __pow__(self, other: Any) -> Matrix:
        """Matrix power (integer powers only)."""
        if isinstance(other, int):
            if self.shape[0] != self.shape[1]:
                raise ValueError(
                    "Matrix power only defined for square matrices")

            if other == 0:
                # Identity matrix
                n = self.shape[0]
                identity = [
                    [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
                return Matrix(identity)
            elif other > 0:
                result = self
                for _ in range(other - 1):
                    result = result * self
                return result
            else:
                # Negative power: inverse first
                return self.inverse() ** (-other)
        else:
            return NotImplemented

    def __rpow__(self, other: Any) -> MathValue:
        """Right power not supported."""
        raise TypeError("Matrix does not support right exponentiation")

    def __neg__(self) -> Matrix:
        """Negation."""
        return Matrix([[-el for el in row] for row in self.rows])

    def __pos__(self) -> Matrix:
        """Unary positive."""
        return Matrix([[+el for el in row] for row in self.rows])

    def __abs__(self) -> MathValue:
        """Absolute value not well-defined for matrices."""
        raise TypeError(
            "Absolute value not defined for matrices (use norm or determinant)")


# Standalone functions (Perl-style interface)

def norm(obj: Vector | Point) -> Real:
    """
    Calculate the Euclidean norm (magnitude) of a vector or point.

    This is a standalone function matching Perl's norm() function.

    Args:
        obj: Vector or Point object

    Returns:
        Real number representing the norm

    Examples:
        >>> v = Vector(3, 4)
        >>> norm(v)
        Real(5.0)

    Reference: lib/Value/Vector.pm::norm
    """
    if isinstance(obj, (Vector, Point)):
        return obj.norm()
    else:
        raise TypeError(f"norm() requires a Vector or Point, got {type(obj)}")


