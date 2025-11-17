"""Tests for Point MathValue class."""

import pytest

from pg.math.geometric import Point
from pg.math.numeric import Real
from pg.math.value import MathValue


class TestPointBasicInstantiation:
    """Test basic instantiation and field access."""

    def test_instantiate_simple_point_2d(self):
        """Test creating a 2D point with coordinates."""
        p = Point(1, 2)
        assert len(p) == 2
        assert p[0].value == 1.0
        assert p[1].value == 2.0

    def test_instantiate_simple_point_3d(self):
        """Test creating a 3D point."""
        p = Point(1, 2, 3)
        assert len(p) == 3
        assert p[0].value == 1.0
        assert p[1].value == 2.0
        assert p[2].value == 3.0

    def test_instantiate_from_list(self):
        """Test creating point from list."""
        p = Point([1, 2, 3])
        assert len(p) == 3
        assert p[0].value == 1.0

    def test_instantiate_from_tuple(self):
        """Test creating point from tuple."""
        p = Point((4, 5, 6))
        assert len(p) == 3
        assert p[0].value == 4.0

    def test_instantiate_single_coordinate(self):
        """Test creating a 1D point."""
        p = Point(5)
        assert len(p) == 1
        assert p[0].value == 5.0

    def test_instantiate_negative_coordinates(self):
        """Test point with negative coordinates."""
        p = Point(-1, -2)
        assert len(p) == 2
        assert p[0].value == -1.0
        assert p[1].value == -2.0

    def test_instantiate_float_coordinates(self):
        """Test point with float coordinates."""
        p = Point(1.5, 2.7)
        assert len(p) == 2
        assert abs(p[0].value - 1.5) < 1e-10
        assert abs(p[1].value - 2.7) < 1e-10

    def test_instantiate_with_real_objects(self):
        """Test creating point from Real objects."""
        r1 = Real(3)
        r2 = Real(4)
        p = Point(r1, r2)
        assert len(p) == 2
        assert p[0].value == 3.0

    def test_point_type_precedence(self):
        """Test that Point has correct type precedence."""
        p = Point(1, 2)
        assert hasattr(p, 'type_precedence')

    def test_point_context_optional(self):
        """Test that Point accepts context parameter."""
        p = Point(1, 2, context=None)
        assert p is not None


class TestPointStringParsing:
    """Test string parsing for point creation."""

    def test_parse_string_point_2d(self):
        """Test parsing point string '(1, 2)'."""
        p = Point("(1, 2)")
        assert len(p) == 2
        # String parsing creates String objects for coordinates
        assert p[0].value == "1"
        assert p[1].value == "2"

    def test_parse_string_point_3d(self):
        """Test parsing 3D point string."""
        p = Point("(1, 2, 3)")
        assert len(p) == 3
        # String parsing creates String objects
        assert p[0].value == "1"

    def test_parse_string_point_with_spaces(self):
        """Test parsing point string with extra spaces."""
        p = Point("( 1 , 2 , 3 )")
        assert len(p) == 3

    def test_parse_string_point_negative(self):
        """Test parsing point with negative coordinates."""
        p = Point("(-1, -2)")
        assert len(p) == 2
        # String parsing creates String objects
        assert p[0].value == "-1"
        assert p[1].value == "-2"

    def test_parse_string_point_floats(self):
        """Test parsing point with float coordinates."""
        p = Point("(1.5, 2.7)")
        assert len(p) == 2


class TestPointConversions:
    """Test conversion methods."""

    def test_to_string_2d(self):
        """Test to_string() for 2D point."""
        p = Point(1, 2)
        result = p.to_string()
        assert "1" in result
        assert "2" in result
        assert "(" in result
        assert ")" in result

    def test_to_string_3d(self):
        """Test to_string() for 3D point."""
        p = Point(1, 2, 3)
        result = p.to_string()
        assert "1" in result
        assert "2" in result
        assert "3" in result

    def test_to_string_single_coordinate(self):
        """Test to_string() for 1D point."""
        p = Point(5)
        result = p.to_string()
        assert "5" in result

    def test_to_tex_format(self):
        """Test to_tex() returns LaTeX format."""
        p = Point(1, 2)
        result = p.to_tex()
        assert isinstance(result, str)
        assert "(" in result or "left" in result.lower()

    def test_to_python_returns_tuple(self):
        """Test to_python() returns tuple."""
        p = Point(1, 2, 3)
        result = p.to_python()
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0] == 1.0
        assert result[1] == 2.0
        assert result[2] == 3.0

    def test_to_python_2d(self):
        """Test to_python() for 2D point."""
        p = Point(3.5, 4.5)
        result = p.to_python()
        assert result == (3.5, 4.5)


class TestPointIndexing:
    """Test coordinate access via indexing."""

    def test_getitem_first_coordinate(self):
        """Test accessing first coordinate."""
        p = Point(1, 2, 3)
        coord = p[0]
        assert isinstance(coord, MathValue)
        assert coord.value == 1.0

    def test_getitem_last_coordinate(self):
        """Test accessing last coordinate."""
        p = Point(1, 2, 3)
        coord = p[2]
        assert coord.value == 3.0

    def test_getitem_negative_index(self):
        """Test negative indexing."""
        p = Point(1, 2, 3)
        coord = p[-1]
        assert coord.value == 3.0

    def test_getitem_out_of_bounds_raises_error(self):
        """Test that out of bounds access raises IndexError."""
        p = Point(1, 2)
        with pytest.raises(IndexError):
            _ = p[5]

    def test_len_returns_dimension(self):
        """Test that len() returns dimension."""
        p = Point(1, 2, 3)
        assert len(p) == 3


class TestPointComparison:
    """Test comparison operations."""

    def test_eq_identical_points(self):
        """Test equality of identical points."""
        p1 = Point(1, 2)
        p2 = Point(1, 2)
        assert p1 == p2

    def test_eq_points_different_order(self):
        """Test inequality when coordinates differ."""
        p1 = Point(1, 2)
        p2 = Point(2, 1)
        assert p1 != p2

    def test_compare_method_exact_match(self):
        """Test compare() method with exact match."""
        p1 = Point(1, 2)
        p2 = Point(1, 2)
        assert p1.compare(p2) is True

    def test_compare_method_different(self):
        """Test compare() method with different values."""
        p1 = Point(1, 2)
        p2 = Point(1, 3)
        assert p1.compare(p2) is False

    def test_compare_with_tolerance(self):
        """Test compare with tolerance."""
        p1 = Point(1.0, 2.0)
        p2 = Point(1.0005, 2.0005)
        assert p1.compare(p2, tolerance=0.001) is True

    def test_compare_different_dimensions(self):
        """Test that points with different dimensions don't compare equal."""
        p1 = Point(1, 2)
        p2 = Point(1, 2, 3)
        assert p1.compare(p2) is False


class TestPointDistance:
    """Test distance calculations."""

    def test_distance_to_same_point(self):
        """Test distance to same point is zero."""
        p1 = Point(1, 2)
        p2 = Point(1, 2)
        dist = p1.distance(p2)
        assert abs(dist.value) < 1e-10

    def test_distance_2d_pythagorean(self):
        """Test 2D distance using Pythagorean theorem."""
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        dist = p1.distance(p2)
        assert abs(dist.value - 5.0) < 1e-10

    def test_distance_3d(self):
        """Test 3D distance."""
        p1 = Point(0, 0, 0)
        p2 = Point(1, 0, 0)
        dist = p1.distance(p2)
        assert abs(dist.value - 1.0) < 1e-10

    def test_distance_negative_coordinates(self):
        """Test distance with negative coordinates."""
        p1 = Point(-1, -1)
        p2 = Point(2, 3)
        # distance = sqrt((2-(-1))^2 + (3-(-1))^2) = sqrt(9 + 16) = 5
        dist = p1.distance(p2)
        assert abs(dist.value - 5.0) < 1e-10

    def test_distance_returns_real(self):
        """Test that distance returns Real object."""
        p1 = Point(0, 0)
        p2 = Point(1, 1)
        dist = p1.distance(p2)
        assert isinstance(dist, Real)


class TestPointEdgeCases:
    """Test edge cases and special points."""

    def test_origin_point(self):
        """Test origin point (0, 0)."""
        p = Point(0, 0)
        assert len(p) == 2
        assert p[0].value == 0.0
        assert p[1].value == 0.0

    def test_very_large_coordinates(self):
        """Test point with very large coordinates."""
        p = Point(1e10, 2e10)
        assert p[0].value == 1e10

    def test_very_small_coordinates(self):
        """Test point with very small coordinates."""
        p = Point(1e-10, 2e-10)
        assert p[0].value == 1e-10

    def test_high_dimensional_point(self):
        """Test point in high dimensions."""
        coords = list(range(10))
        p = Point(*coords)
        assert len(p) == 10
        assert p[9].value == 9.0

    def test_is_instance_of_mathvalue(self):
        """Test that Point is instance of MathValue."""
        p = Point(1, 2)
        assert isinstance(p, MathValue)


class TestPointIntegration:
    """Integration tests for Point class."""

    def test_workflow_create_and_distance(self):
        """Test workflow: create points and calculate distance."""
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        dist = p1.distance(p2)
        assert isinstance(dist, Real)
        assert abs(dist.value - 5.0) < 1e-10

    def test_workflow_string_parse_and_access(self):
        """Test workflow: parse string and access coordinates."""
        p = Point("(1, 2, 3)")
        assert len(p) == 3
        # String parsing creates String objects
        assert p[0].value == "1"
        result = p.to_string()
        assert "1" in result and "2" in result and "3" in result

    def test_workflow_coordinate_conversion(self):
        """Test workflow: create point and convert."""
        p = Point(1, 2)
        coords = p.to_python()
        assert isinstance(coords, tuple)
        assert coords == (1.0, 2.0)

    def test_workflow_triangle_distances(self):
        """Test workflow: triangle with three points."""
        p1 = Point(0, 0)
        p2 = Point(3, 0)
        p3 = Point(0, 4)
        d12 = p1.distance(p2)
        d23 = p2.distance(p3)
        d31 = p3.distance(p1)
        assert abs(d12.value - 3.0) < 1e-10
        assert abs(d23.value - 5.0) < 1e-10
        assert abs(d31.value - 4.0) < 1e-10
