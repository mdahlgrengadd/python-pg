"""Tests for Interval MathValue class."""

import pytest

from pg.math.sets import Interval, Union
from pg.math.numeric import Real, Infinity


class TestIntervalConstruction:
    """Verify the various constructor signatures."""

    def test_construct_from_string_literal(self):
        interval = Interval("(1,5)")
        assert isinstance(interval.left, Real)
        assert interval.left.value == 1.0
        assert interval.open_left is True
        assert interval.open_right is True

    def test_construct_from_bracket_arguments(self):
        interval = Interval('[', 2, 4, ']')
        assert interval.open_left is False
        assert interval.open_right is False
        assert interval.left.value == 2.0
        assert interval.right.value == 4.0

    def test_construct_from_booleans(self):
        interval = Interval(0, 10, False, True)
        assert interval.open_left is False
        assert interval.open_right is True

    def test_construct_defaults_to_open_interval(self):
        interval = Interval(-1, 1)
        assert interval.open_left is True
        assert interval.open_right is True

    def test_invalid_string_literal_raises(self):
        with pytest.raises(ValueError):
            Interval("invalid")

    def test_invalid_argument_count_raises(self):
        with pytest.raises(ValueError):
            Interval(1)


class TestIntervalBehaviors:
    """Test Interval behavior such as contains and compare."""

    def test_contains_respects_open_and_closed_endpoints(self):
        closed = Interval('[', 0, 2, ']')
        open_interval = Interval('(0,2)')
        assert closed.contains(0) is True
        assert open_interval.contains(0) is False
        assert open_interval.contains(1.5) is True
        assert closed.contains(2) is True
        assert open_interval.contains(2) is False

    def test_contains_handles_infinity_endpoints(self):
        interval = Interval('[', -Infinity(-1), Infinity(1), ')')
        assert interval.contains(-1_000_000) is True
        assert interval.contains(1_000_000) is True

    def test_compare_matches_matching_intervals(self):
        first = Interval('[', 0, 5, ')')
        second = Interval(0, 5, False, True)
        assert first.compare(second) is True

    def test_compare_detects_difference(self):
        first = Interval('[', 0, 5, ')')
        second = Interval('(0,5)')
        assert first.compare(second) is False

    def test_is_empty_when_endpoints_meet_and_open(self):
        interval = Interval(1, 1)
        assert interval.is_empty() is True

    def test_length_with_finite_endpoints(self):
        interval = Interval('[', -3, 7, ']')
        length = interval.length()
        assert isinstance(length, Real)
        assert length.value == 10.0

    def test_length_into_infinite_interval(self):
        interval = Interval(-Infinity(-1), 0)
        length = interval.length()
        assert isinstance(length, Infinity)


class TestIntervalOperations:
    """Test higher-level interval operations."""

    def test_intersection_of_overlapping_intervals(self):
        first = Interval('[', 0, 5, ')')
        second = Interval('(2,7)')
        result = first.intersect(second)
        assert isinstance(result, Interval)
        assert result.left.value == 2.0
        assert result.open_left is True
        assert result.right.value == 5.0
        assert result.open_right is True

    def test_intersection_of_disjoint_intervals_returns_none(self):
        first = Interval('[', 0, 1, ']')
        second = Interval('[', 2, 3, ']')
        assert first.intersect(second) is None

    def test_union_merges_overlapping_intervals(self):
        first = Interval('[', 0, 2, ']')
        second = Interval('(1,4]')
        union = first.union(second)
        assert isinstance(union, Union)
        assert len(union.sets) == 1
        merged = union.sets[0]
        assert merged.left.value == 0.0
        assert merged.open_left is False
        assert merged.right.value == 4.0
        assert merged.open_right is False

    def test_union_keeps_disjoint_intervals_separate(self):
        first = Interval('[', 0, 1, ']')
        second = Interval('[', 3, 4, ']')
        union = first.union(second)
        assert len(union.sets) == 2

    def test_to_string_and_to_tex_formats(self):
        interval = Interval('[', 1, 2, ')')
        assert interval.to_string() == "[1, 2)"
        tex = interval.to_tex()
        assert tex.startswith('[') and tex.endswith(')')

    def test_to_python_serializes_tuple(self):
        interval = Interval('(0,5)')
        result = interval.to_python()
        assert result == (0.0, 5.0, True, True)
