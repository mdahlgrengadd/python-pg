"""
Set MathValue types: Interval, Set, Union.

These types represent mathematical sets with set operations.

Reference: lib/Value/Interval.pm, lib/Value/Set.pm, lib/Value/Union.pm
"""

from __future__ import annotations

from typing import Any

from .numeric import Infinity, Real
from .value import MathValue, ToleranceMode, TypePrecedence


class Interval(MathValue):
    """
    Mathematical interval with open/closed endpoints.

    Examples:
    - [0, 1] - closed interval
    - (0, 1) - open interval
    - [0, 1) - half-open interval
    - (-inf, inf) - all real numbers

    Reference: lib/Value/Interval.pm
    """

    type_precedence = TypePrecedence.INTERVAL

    def __init__(self, *args):
        """
        Initialize an Interval.

        Supports Perl-style constructor syntax:
        - Interval('[', 1, 5, ']') - with bracket arguments
        - Interval(1, 5) - defaults to open on both sides
        - Interval(1, 5, False, True) - with boolean open_left, open_right

        Args:
            args: Variable arguments:
                - 4 args: open_bracket, left, right, close_bracket
                - 2 args: left, right (both open)
                - 4 args (last 2 bool): left, right, open_left, open_right

        Reference: lib/Value/Interval.pm::new
        """
        from .value import MathValue as MV

        # Parse arguments based on count and type
        if len(args) == 1 and isinstance(args[0], str):
            # String notation: "(0,5)" or "[2,7]"
            import re
            s = args[0].strip()

            # Match pattern: opening bracket, number, comma, number, closing bracket
            match = re.match(
                r'^([\(\[])(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)([\)\]])$', s)
            if not match:
                raise ValueError(f"Invalid interval string notation: {s}")

            open_bracket = match.group(1)
            left_val = float(match.group(2))
            right_val = float(match.group(3))
            close_bracket = match.group(4)

            open_left = (open_bracket == '(')
            open_right = (close_bracket == ')')
        elif len(args) == 4:
            # Check if first and last args are bracket strings
            if isinstance(args[0], str) and isinstance(args[3], str):
                # Perl-style: Interval('[', 1, 5, ']')
                open_bracket = args[0]
                left_val = args[1]
                right_val = args[2]
                close_bracket = args[3]

                # Convert brackets to boolean flags
                open_left = (open_bracket == '(')
                open_right = (close_bracket == ')')
            elif isinstance(args[2], bool) and isinstance(args[3], bool):
                # Python-style: Interval(1, 5, False, True)
                left_val = args[0]
                right_val = args[1]
                open_left = args[2]
                open_right = args[3]
            else:
                raise ValueError("Invalid Interval constructor arguments")
        elif len(args) == 2:
            # Interval(1, 5) - defaults to open on both sides
            left_val = args[0]
            right_val = args[1]
            open_left = True
            open_right = True
        else:
            raise ValueError(
                f"Interval requires 2 or 4 arguments, got {len(args)}")

        # Convert to MathValue
        self.left = MV.from_python(left_val) if not isinstance(
            left_val, MathValue) else left_val
        self.right = MV.from_python(right_val) if not isinstance(
            right_val, MathValue) else right_val
        self.open_left = open_left
        self.open_right = open_right

        # Validate interval
        if not isinstance(self.left, (Real, Infinity)) or not isinstance(
            self.right, (Real, Infinity)
        ):
            raise TypeError("Interval endpoints must be Real or Infinity")

    def promote(self, other: MathValue) -> MathValue:
        """Intervals don't promote to other types."""
        return self

    def compare(
        self, other: MathValue, tolerance: float = 0.001, mode: str = ToleranceMode.RELATIVE
    ) -> bool:
        """Compare intervals."""
        if not isinstance(other, Interval):
            return False

        # Compare endpoints and openness
        return (
            self.left.compare(other.left, tolerance, mode)
            and self.right.compare(other.right, tolerance, mode)
            and self.open_left == other.open_left
            and self.open_right == other.open_right
        )

    def cmp(self, **options):
        """
        Create answer evaluator for this interval.

        Returns:
            Answer evaluator that checks if student's answer matches this interval
        """
        from pg.answer.evaluators.string import StringEvaluator

        # For now, use string comparison with the interval notation
        interval_str = str(self)
        return StringEvaluator(
            correct_answer=interval_str,
            case_sensitive=False,
            trim_whitespace=True,
            **options,
        )

    def contains(self, value: MathValue | float) -> bool:
        """
        Check if a value is in the interval.

        Args:
            value: Value to check

        Returns:
            True if value is in the interval
        """
        from .value import MathValue as MV

        val = MV.from_python(value) if not isinstance(
            value, MathValue) else value

        if not isinstance(val, Real):
            return False

        v = val.value
        left_val = self.left.value if isinstance(
            self.left, Real) else float("-inf")
        right_val = self.right.value if isinstance(
            self.right, Real) else float("inf")

        # Check left boundary
        if self.open_left:
            if v <= left_val:
                return False
        else:
            if v < left_val:
                return False

        # Check right boundary
        if self.open_right:
            if v >= right_val:
                return False
        else:
            if v > right_val:
                return False

        return True

    def is_empty(self) -> bool:
        """Check if interval is empty."""
        if isinstance(self.left, Real) and isinstance(self.right, Real):
            if self.left.value > self.right.value:
                return True
            if self.left.value == self.right.value and (self.open_left or self.open_right):
                return True
        return False

    def length(self) -> Real | Infinity:
        """
        Calculate the length of the interval.

        Returns:
            Real number or Infinity
        """
        if self.is_empty():
            return Real(0.0)

        if isinstance(self.left, Infinity) or isinstance(self.right, Infinity):
            return Infinity(1)

        return Real(self.right.value - self.left.value)

    def intersect(self, other: Interval) -> Interval | None:
        """
        Compute the intersection of two intervals.

        Args:
            other: Another interval

        Returns:
            Intersection interval, or None if empty
        """
        # Determine new left endpoint (maximum of lefts)
        left_val = (
            self.left.value
            if isinstance(self.left, Real)
            else float("-inf")
        )
        other_left_val = (
            other.left.value
            if isinstance(other.left, Real)
            else float("-inf")
        )

        if left_val > other_left_val:
            new_left = self.left
            new_open_left = self.open_left
        elif left_val < other_left_val:
            new_left = other.left
            new_open_left = other.open_left
        else:
            # Equal: open if either is open
            new_left = self.left
            new_open_left = self.open_left or other.open_left

        # Determine new right endpoint (minimum of rights)
        right_val = (
            self.right.value
            if isinstance(self.right, Real)
            else float("inf")
        )
        other_right_val = (
            other.right.value
            if isinstance(other.right, Real)
            else float("inf")
        )

        if right_val < other_right_val:
            new_right = self.right
            new_open_right = self.open_right
        elif right_val > other_right_val:
            new_right = other.right
            new_open_right = other.open_right
        else:
            # Equal: open if either is open
            new_right = self.right
            new_open_right = self.open_right or other.open_right

        result = Interval(new_left, new_right, new_open_left, new_open_right)
        return result if not result.is_empty() else None

    def union(self, other: Interval) -> Union:
        """
        Compute the union of two intervals.

        Args:
            other: Another interval

        Returns:
            Union object (may be simplified to single interval if overlapping)
        """
        # Check if intervals overlap or are adjacent
        intersection = self.intersect(other)

        if intersection is not None:
            # Intervals overlap - can merge
            left_val = (
                self.left.value
                if isinstance(self.left, Real)
                else float("-inf")
            )
            other_left_val = (
                other.left.value
                if isinstance(other.left, Real)
                else float("-inf")
            )

            if left_val < other_left_val:
                new_left = self.left
                new_open_left = self.open_left
            elif left_val > other_left_val:
                new_left = other.left
                new_open_left = other.open_left
            else:
                new_left = self.left
                new_open_left = self.open_left and other.open_left

            right_val = (
                self.right.value
                if isinstance(self.right, Real)
                else float("inf")
            )
            other_right_val = (
                other.right.value
                if isinstance(other.right, Real)
                else float("inf")
            )

            if right_val > other_right_val:
                new_right = self.right
                new_open_right = self.open_right
            elif right_val < other_right_val:
                new_right = other.right
                new_open_right = other.open_right
            else:
                new_right = self.right
                new_open_right = self.open_right and other.open_right

            merged = Interval(new_left, new_right,
                              new_open_left, new_open_right)
            return Union([merged])
        else:
            # Disjoint intervals
            return Union([self, other])

    def to_string(self) -> str:
        """Convert to string."""
        left_bracket = "(" if self.open_left else "["
        right_bracket = ")" if self.open_right else "]"
        return f"{left_bracket}{self.left.to_string()}, {self.right.to_string()}{right_bracket}"

    def to_tex(self) -> str:
        """Convert to LaTeX."""
        left_bracket = "(" if self.open_left else "["
        right_bracket = ")" if self.open_right else "]"
        return f"{left_bracket}{self.left.to_tex()}, {self.right.to_tex()}{right_bracket}"

    def to_python(self) -> tuple[float, float, bool, bool]:
        """Convert to Python tuple."""
        return (
            self.left.to_python(),
            self.right.to_python(),
            self.open_left,
            self.open_right,
        )

    # Arithmetic operators (not standard for intervals, but sometimes useful)

    def __add__(self, other: Any) -> MathValue:
        """Addition not well-defined for intervals."""
        raise TypeError("Interval does not support addition")

    def __radd__(self, other: Any) -> MathValue:
        """Right addition not supported."""
        raise TypeError("Interval does not support addition")

    def __sub__(self, other: Any) -> MathValue:
        """Subtraction not supported."""
        raise TypeError("Interval does not support subtraction")

    def __rsub__(self, other: Any) -> MathValue:
        """Right subtraction not supported."""
        raise TypeError("Interval does not support subtraction")

    def __mul__(self, other: Any) -> MathValue:
        """Multiplication not supported."""
        raise TypeError("Interval does not support multiplication")

    def __rmul__(self, other: Any) -> MathValue:
        """Right multiplication not supported."""
        raise TypeError("Interval does not support multiplication")

    def __truediv__(self, other: Any) -> MathValue:
        """Division not supported."""
        raise TypeError("Interval does not support division")

    def __rtruediv__(self, other: Any) -> MathValue:
        """Right division not supported."""
        raise TypeError("Interval does not support division")

    def __pow__(self, other: Any) -> MathValue:
        """Power not supported."""
        raise TypeError("Interval does not support exponentiation")

    def __rpow__(self, other: Any) -> MathValue:
        """Right power not supported."""
        raise TypeError("Interval does not support exponentiation")

    def __neg__(self) -> MathValue:
        """Negation not well-defined."""
        raise TypeError("Interval does not support negation")

    def __pos__(self) -> MathValue:
        """Unary positive not supported."""
        raise TypeError("Interval does not support unary positive")

    def __abs__(self) -> MathValue:
        """Absolute value not well-defined."""
        raise TypeError("Interval does not support absolute value")


class Set(MathValue):
    """
    Finite set of elements.

    Examples:
    - {1, 2, 3}
    - {a, b, c}

    Reference: lib/Value/Set.pm
    """

    type_precedence = TypePrecedence.SET

    def __init__(self, elements: list[MathValue] | list[float]):
        """
        Initialize a Set.

        Args:
            elements: List of elements (duplicates will be removed)
        """
        from .value import MathValue as MV

        # Convert to MathValue and remove duplicates
        converted = [MV.from_python(e) if not isinstance(
            e, MathValue) else e for e in elements]

        # Remove duplicates (using Python set with tuple conversion for hashability)
        unique = []
        seen = []
        for elem in converted:
            is_duplicate = False
            for seen_elem in seen:
                if elem.compare(seen_elem):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(elem)
                seen.append(elem)

        self.elements = unique

    def promote(self, other: MathValue) -> MathValue:
        """Sets don't promote."""
        return self

    def compare(
        self, other: MathValue, tolerance: float = 0.001, mode: str = ToleranceMode.RELATIVE
    ) -> bool:
        """Compare sets (same elements, order doesn't matter)."""
        if not isinstance(other, Set):
            return False

        if len(self.elements) != len(other.elements):
            return False

        # Check each element in self is in other
        for elem in self.elements:
            found = False
            for other_elem in other.elements:
                if elem.compare(other_elem, tolerance, mode):
                    found = True
                    break
            if not found:
                return False

        return True

    def contains(self, value: MathValue | float) -> bool:
        """
        Check if a value is in the set.

        Args:
            value: Value to check

        Returns:
            True if value is in the set
        """
        from .value import MathValue as MV

        val = MV.from_python(value) if not isinstance(
            value, MathValue) else value

        for elem in self.elements:
            if val.compare(elem):
                return True
        return False

    def cardinality(self) -> int:
        """Return the number of elements in the set."""
        return len(self.elements)

    def is_empty(self) -> bool:
        """Check if set is empty."""
        return len(self.elements) == 0

    def intersect(self, other: Set) -> Set:
        """
        Compute the intersection of two sets.

        Args:
            other: Another set

        Returns:
            New set containing elements in both sets
        """
        result = []
        for elem in self.elements:
            if other.contains(elem):
                result.append(elem)
        return Set(result)

    def union(self, other: Set) -> Set:
        """
        Compute the union of two sets.

        Args:
            other: Another set

        Returns:
            New set containing elements from both sets
        """
        # Start with all elements from self
        result = list(self.elements)

        # Add elements from other that aren't already in result
        for elem in other.elements:
            if not self.contains(elem):
                result.append(elem)

        return Set(result)

    def difference(self, other: Set) -> Set:
        """
        Compute the set difference (self - other).

        Args:
            other: Another set

        Returns:
            New set containing elements in self but not in other
        """
        result = []
        for elem in self.elements:
            if not other.contains(elem):
                result.append(elem)
        return Set(result)

    def is_subset(self, other: Set) -> bool:
        """Check if this set is a subset of another."""
        for elem in self.elements:
            if not other.contains(elem):
                return False
        return True

    def is_superset(self, other: Set) -> bool:
        """Check if this set is a superset of another."""
        return other.is_subset(self)

    def to_string(self) -> str:
        """Convert to string."""
        if self.is_empty():
            return "{}"
        elements_str = ", ".join(e.to_string() for e in self.elements)
        return f"{{{elements_str}}}"

    def to_tex(self) -> str:
        """Convert to LaTeX."""
        if self.is_empty():
            return "\\emptyset"
        elements_str = ", ".join(e.to_tex() for e in self.elements)
        return f"\\{{{elements_str}\\}}"

    def to_python(self) -> list[Any]:
        """Convert to Python list."""
        return [e.to_python() for e in self.elements]

    # Arithmetic operators (not standard for sets)

    def __add__(self, other: Any) -> MathValue:
        """Addition not supported."""
        raise TypeError("Set does not support addition")

    def __radd__(self, other: Any) -> MathValue:
        """Right addition not supported."""
        raise TypeError("Set does not support addition")

    def __sub__(self, other: Any) -> MathValue:
        """Subtraction not supported."""
        raise TypeError("Set does not support subtraction")

    def __rsub__(self, other: Any) -> MathValue:
        """Right subtraction not supported."""
        raise TypeError("Set does not support subtraction")

    def __mul__(self, other: Any) -> MathValue:
        """Multiplication not supported."""
        raise TypeError("Set does not support multiplication")

    def __rmul__(self, other: Any) -> MathValue:
        """Right multiplication not supported."""
        raise TypeError("Set does not support multiplication")

    def __truediv__(self, other: Any) -> MathValue:
        """Division not supported."""
        raise TypeError("Set does not support division")

    def __rtruediv__(self, other: Any) -> MathValue:
        """Right division not supported."""
        raise TypeError("Set does not support division")

    def __pow__(self, other: Any) -> MathValue:
        """Power not supported."""
        raise TypeError("Set does not support exponentiation")

    def __rpow__(self, other: Any) -> MathValue:
        """Right power not supported."""
        raise TypeError("Set does not support exponentiation")

    def __neg__(self) -> MathValue:
        """Negation not supported."""
        raise TypeError("Set does not support negation")

    def __pos__(self) -> MathValue:
        """Unary positive not supported."""
        raise TypeError("Set does not support unary positive")

    def __abs__(self) -> MathValue:
        """Absolute value not supported."""
        raise TypeError("Set does not support absolute value")


class Union(MathValue):
    """
    Union of intervals and/or sets.

    Examples:
    - [0, 1] ∪ [2, 3]
    - (-inf, 0] ∪ [1, inf)

    Reference: lib/Value/Union.pm
    """

    type_precedence = TypePrecedence.UNION

    def __init__(self, sets: list[Interval | Set] | str):
        """
        Initialize a Union.

        Args:
            sets: List of Intervals and/or Sets, or string notation like "(0,2) U [5,7]"
        """
        # Handle string notation
        if isinstance(sets, str):
            import re
            # Split by 'U' or 'u' (union symbol)
            parts = re.split(r'\s*[Uu]\s*', sets.strip())
            parsed_sets = []
            for part in parts:
                part = part.strip()
                # Try to parse each part as an Interval
                try:
                    parsed_sets.append(Interval(part))
                except (ValueError, TypeError):
                    # If not an interval, might be a set - skip for now
                    pass
            sets = parsed_sets

        if not all(isinstance(s, (Interval, Set)) for s in sets):
            raise TypeError("Union elements must be Intervals or Sets")

        self.sets = sets
        self._simplify()

    def _simplify(self) -> None:
        """Simplify the union by merging overlapping intervals."""
        # Separate intervals and sets
        intervals = [s for s in self.sets if isinstance(s, Interval)]
        sets = [s for s in self.sets if isinstance(s, Set)]

        # Try to merge intervals iteratively until no more merges possible
        if len(intervals) > 1:
            changed = True
            while changed:
                changed = False
                merged = []
                used = set()

                for i, interval1 in enumerate(intervals):
                    if i in used:
                        continue

                    current = interval1
                    for j in range(i + 1, len(intervals)):
                        if j in used:
                            continue

                        interval2 = intervals[j]
                        intersection = current.intersect(interval2)
                        if intersection is not None:
                            # Intervals overlap - merge them
                            merged_union = current.union(interval2)
                            if len(merged_union.sets) == 1:
                                current = merged_union.sets[0]
                                used.add(j)
                                changed = True

                    merged.append(current)
                    used.add(i)

                intervals = merged

        self.sets = intervals + sets

    def promote(self, other: MathValue) -> MathValue:
        """Unions don't promote."""
        return self

    def compare(
        self, other: MathValue, tolerance: float = 0.001, mode: str = ToleranceMode.RELATIVE
    ) -> bool:
        """Compare unions (same constituent sets)."""
        if not isinstance(other, Union):
            return False

        if len(self.sets) != len(other.sets):
            return False

        # Each set in self must match a set in other
        for s in self.sets:
            found = False
            for other_s in other.sets:
                if s.compare(other_s, tolerance, mode):
                    found = True
                    break
            if not found:
                return False

        return True

    def contains(self, value: MathValue | float) -> bool:
        """
        Check if a value is in the union.

        Args:
            value: Value to check

        Returns:
            True if value is in any of the constituent sets
        """
        for s in self.sets:
            if s.contains(value):
                return True
        return False

    def to_string(self) -> str:
        """Convert to string."""
        if len(self.sets) == 0:
            return "{}"
        return " U ".join(s.to_string() for s in self.sets)

    def to_tex(self) -> str:
        """Convert to LaTeX."""
        if len(self.sets) == 0:
            return "\\emptyset"
        return " \\cup ".join(s.to_tex() for s in self.sets)

    def to_python(self) -> list[Any]:
        """Convert to Python list of constituent sets."""
        return [s.to_python() for s in self.sets]

    # Arithmetic operators (not standard for unions)

    def __add__(self, other: Any) -> MathValue:
        """Addition not supported."""
        raise TypeError("Union does not support addition")

    def __radd__(self, other: Any) -> MathValue:
        """Right addition not supported."""
        raise TypeError("Union does not support addition")

    def __sub__(self, other: Any) -> MathValue:
        """Subtraction not supported."""
        raise TypeError("Union does not support subtraction")

    def __rsub__(self, other: Any) -> MathValue:
        """Right subtraction not supported."""
        raise TypeError("Union does not support subtraction")

    def __mul__(self, other: Any) -> MathValue:
        """Multiplication not supported."""
        raise TypeError("Union does not support multiplication")

    def __rmul__(self, other: Any) -> MathValue:
        """Right multiplication not supported."""
        raise TypeError("Union does not support multiplication")

    def __truediv__(self, other: Any) -> MathValue:
        """Division not supported."""
        raise TypeError("Union does not support division")

    def __rtruediv__(self, other: Any) -> MathValue:
        """Right division not supported."""
        raise TypeError("Union does not support division")

    def __pow__(self, other: Any) -> MathValue:
        """Power not supported."""
        raise TypeError("Union does not support exponentiation")

    def __rpow__(self, other: Any) -> MathValue:
        """Right power not supported."""
        raise TypeError("Union does not support exponentiation")

    def __neg__(self) -> MathValue:
        """Negation not supported."""
        raise TypeError("Union does not support negation")

    def __pos__(self) -> MathValue:
        """Unary positive not supported."""
        raise TypeError("Union does not support unary positive")

    def __abs__(self) -> MathValue:
        """Absolute value not supported."""
        raise TypeError("Union does not support absolute value")
