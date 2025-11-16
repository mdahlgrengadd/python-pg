"""Tests for Set and Union MathValue classes."""

import pytest

from pg.math.numeric import Real
from pg.math.sets import Interval, Set, Union

class TestSetConstruction:
    def test_construct_from_python_numbers_deduplicates(self):
        s = Set([1, 2, 2, 3])
        assert s.cardinality() == 3
        assert s.contains(2) is True

    def test_construct_from_mathvalues(self):
        elements = [Real(1), Real(2)]
        s = Set(elements)
        assert all(isinstance(elem, Real) for elem in s.elements)

    def test_compare_is_order_independent(self):
        s1 = Set([1, 2, 3])
        s2 = Set([3, 2, 1])
        assert s1.compare(s2) is True

    def test_compare_detects_different_sets(self):
        s1 = Set([1, 2])
        s2 = Set([1, 3])
        assert s1.compare(s2) is False

    def test_contains_handles_nonexistent_element(self):
        s = Set([1, 2])
        assert s.contains(3) is False

    def test_is_empty_and_cardinality(self):
        empty = Set([])
        assert empty.is_empty() is True
        assert empty.cardinality() == 0

class TestSetOperations:
    def test_union_combines_unique_elements(self):
        s1 = Set([1, 2])
        s2 = Set([2, 3])
        result = s1.union(s2)
        assert result.cardinality() == 3
        assert result.contains(3) is True

    def test_intersection_keeps_common_elements(self):
        s1 = Set([1, 2, 3])
        s2 = Set([2, 4])
        result = s1.intersect(s2)
        assert result.cardinality() == 1
        assert result.contains(2) is True

    def test_difference_removes_other_elements(self):
        s1 = Set([1, 2, 3])
        s2 = Set([2])
        result = s1.difference(s2)
        assert result.cardinality() == 2
        assert result.contains(2) is False

    def test_subset_and_superset_checks(self):
        subset = Set([1, 2])
        superset = Set([1, 2, 3])
        assert subset.is_subset(superset) is True
        assert superset.is_superset(subset) is True

    def test_to_string_and_to_tex_formats(self):
        s = Set([1, 2])
        assert s.to_string() == "{1, 2}"
        tex = s.to_tex()
        assert tex.startswith('\\{') and tex.endswith('\\}')

class TestUnionBehaviors:
    def test_union_from_intervals_merges_overlap(self):
        interval1 = Interval('[', 0, 2, ']')
        interval2 = Interval('(1,4]')
        union = Union([interval1, interval2])
        assert len(union.sets) == 1
        merged = union.sets[0]
        assert merged.left.value == 0.0
        assert merged.open_left is False
        assert merged.right.value == 4.0

    def test_union_from_string_notation(self):
        union = Union("(0,1] U [2,3]")
        assert len(union.sets) == 2
        assert all(isinstance(s, Interval) for s in union.sets)

    def test_union_contains_checks_member_sets(self):
        interval = Interval('(0,2)')
        set_part = Set([3])
        union = Union([interval, set_part])
        assert union.contains(1) is True
        assert union.contains(3) is True
        assert union.contains(5) is False

    def test_union_compare(self):
        u1 = Union([Interval('(0,1)'), Set([2])])
        u2 = Union([Set([2]), Interval('(0,1)')])
        assert u1.compare(u2) is True

    def test_union_to_string_and_to_tex(self):
        union = Union([Interval('(0,1)'), Set([2])])
        assert "U" in union.to_string()
        assert "\\cup" in union.to_tex()
