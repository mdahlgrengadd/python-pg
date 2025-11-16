"""Tests for Vector MathValue class."""

import pytest
import math

from pg.math.geometric import Vector, Point
from pg.math.numeric import Real
from pg.math.value import MathValue


class TestVectorBasicInstantiation:
    """Test basic instantiation and field access."""

    def test_instantiate_2d_vector(self):
        """Test creating a 2D vector."""
        v = Vector(1, 2)
        assert len(v) == 2
        assert v[0].value == 1.0
        assert v[1].value == 2.0

    def test_instantiate_3d_vector(self):
        """Test creating a 3D vector."""
        v = Vector(1, 2, 3)
        assert len(v) == 3
        assert v[0].value == 1.0
        assert v[1].value == 2.0
        assert v[2].value == 3.0

    def test_instantiate_from_list(self):
        """Test creating vector from list."""
        v = Vector([1, 2, 3])
        assert len(v) == 3
        assert v[0].value == 1.0

    def test_instantiate_from_tuple(self):
        """Test creating vector from tuple."""
        v = Vector((4, 5, 6))
        assert len(v) == 3
        assert v[0].value == 4.0

    def test_instantiate_1d_vector(self):
        """Test creating a 1D vector."""
        v = Vector(5)
        assert len(v) == 1
        assert v[0].value == 5.0

    def test_instantiate_negative_components(self):
        """Test vector with negative components."""
        v = Vector(-1, -2, -3)
        assert len(v) == 3
        assert v[0].value == -1.0

    def test_instantiate_float_components(self):
        """Test vector with float components."""
        v = Vector(1.5, 2.7, 3.9)
        assert len(v) == 3
        assert abs(v[0].value - 1.5) < 1e-10

    def test_instantiate_with_real_objects(self):
        """Test creating vector from Real objects."""
        r1 = Real(3)
        r2 = Real(4)
        v = Vector(r1, r2)
        assert len(v) == 2
        assert v[0].value == 3.0

    def test_vector_type_precedence(self):
        """Test that Vector has correct type precedence."""
        v = Vector(1, 2)
        assert hasattr(v, 'type_precedence')

    def test_vector_len_for_dimension(self):
        """Test that len() returns dimension."""
        v = Vector(1, 2, 3)
        assert len(v) == 3


class TestVectorConversions:
    """Test conversion methods."""

    def test_to_string_2d(self):
        """Test to_string() for 2D vector."""
        v = Vector(1, 2)
        result = v.to_string()
        assert "1" in result
        assert "2" in result
        assert "<" in result or "(" in result

    def test_to_string_3d(self):
        """Test to_string() for 3D vector."""
        v = Vector(1, 2, 3)
        result = v.to_string()
        assert "1" in result
        assert "2" in result
        assert "3" in result

    def test_to_tex_format(self):
        """Test to_tex() returns LaTeX format."""
        v = Vector(1, 2)
        result = v.to_tex()
        assert isinstance(result, str)
        # Should have some LaTeX formatting
        assert len(result) > 0

    def test_to_python_returns_list(self):
        """Test to_python() returns list (vector representation)."""
        v = Vector(1, 2, 3)
        result = v.to_python()
        assert isinstance(result, (list, tuple))
        assert len(result) == 3
        assert result == [1.0, 2.0, 3.0] or result == (1.0, 2.0, 3.0)

    def test_to_python_2d(self):
        """Test to_python() for 2D vector."""
        v = Vector(3.5, 4.5)
        result = v.to_python()
        # Vector.to_python() returns list, not tuple
        assert isinstance(result, (list, tuple))


class TestVectorIndexing:
    """Test component access via indexing."""

    def test_getitem_first_component(self):
        """Test accessing first component."""
        v = Vector(1, 2, 3)
        comp = v[0]
        assert isinstance(comp, MathValue)
        assert comp.value == 1.0

    def test_getitem_last_component(self):
        """Test accessing last component."""
        v = Vector(1, 2, 3)
        comp = v[2]
        assert comp.value == 3.0

    def test_getitem_negative_index(self):
        """Test negative indexing."""
        v = Vector(1, 2, 3)
        comp = v[-1]
        assert comp.value == 3.0

    def test_getitem_out_of_bounds_raises_error(self):
        """Test that out of bounds access raises IndexError."""
        v = Vector(1, 2)
        with pytest.raises(IndexError):
            _ = v[5]

    def test_len_returns_dimension(self):
        """Test that len() returns dimension."""
        v = Vector(1, 2, 3)
        assert len(v) == 3


class TestVectorComparison:
    """Test comparison operations."""

    def test_eq_identical_vectors(self):
        """Test equality of identical vectors."""
        v1 = Vector(1, 2)
        v2 = Vector(1, 2)
        assert v1 == v2

    def test_eq_vectors_different_order(self):
        """Test inequality when components differ."""
        v1 = Vector(1, 2)
        v2 = Vector(2, 1)
        assert v1 != v2

    def test_compare_method_exact_match(self):
        """Test compare() method with exact match."""
        v1 = Vector(1, 2)
        v2 = Vector(1, 2)
        assert v1.compare(v2) is True

    def test_compare_method_different(self):
        """Test compare() method with different values."""
        v1 = Vector(1, 2)
        v2 = Vector(1, 3)
        assert v1.compare(v2) is False

    def test_compare_with_tolerance(self):
        """Test compare with tolerance."""
        v1 = Vector(1.0, 2.0)
        v2 = Vector(1.0005, 2.0005)
        assert v1.compare(v2, tolerance=0.001) is True

    def test_compare_different_dimensions(self):
        """Test that vectors with different dimensions don't compare equal."""
        v1 = Vector(1, 2)
        v2 = Vector(1, 2, 3)
        assert v1.compare(v2) is False


class TestVectorArithmetic:
    """Test arithmetic operations."""

    def test_add_two_vectors(self):
        """Test adding two vectors."""
        v1 = Vector(1, 2)
        v2 = Vector(3, 4)
        result = v1 + v2
        assert isinstance(result, Vector)
        assert result[0].value == 4.0
        assert result[1].value == 6.0

    def test_add_two_vectors_only(self):
        """Test adding two vectors."""
        v1 = Vector(1, 2)
        v2 = Vector(3, 4)
        result = v1 + v2
        assert isinstance(result, Vector)
        assert result[0].value == 4.0

    def test_sub_two_vectors(self):
        """Test subtracting two vectors."""
        v1 = Vector(5, 6)
        v2 = Vector(1, 2)
        result = v1 - v2
        assert isinstance(result, Vector)
        assert result[0].value == 4.0
        assert result[1].value == 4.0

    def test_mul_vector_by_vector(self):
        """Test dot product result when multiplying vectors."""
        v1 = Vector(2, 3)
        v2 = Vector(2, 2)
        result = v1 * v2
        assert isinstance(result, Real)
        assert abs(result.value - 10.0) < 1e-10

    def test_div_two_vectors(self):
        """Test dividing by vector raises TypeError (only scalar supported)."""
        v1 = Vector(4, 6)
        v2 = Vector(2, 2)
        with pytest.raises(TypeError):
            _ = v1 / v2


class TestVectorNorm:
    """Test norm (magnitude) calculations."""

    def test_norm_zero_vector(self):
        """Test norm of zero vector."""
        v = Vector(0, 0)
        norm = v.norm()
        assert isinstance(norm, Real)
        assert abs(norm.value) < 1e-10

    def test_norm_unit_vector_x(self):
        """Test norm of unit vector in x direction."""
        v = Vector(1, 0, 0)
        norm = v.norm()
        assert abs(norm.value - 1.0) < 1e-10

    def test_norm_2d_pythagorean(self):
        """Test 2D norm using Pythagorean theorem."""
        v = Vector(3, 4)
        norm = v.norm()
        assert abs(norm.value - 5.0) < 1e-10

    def test_norm_3d(self):
        """Test 3D norm."""
        v = Vector(1, 2, 2)
        norm = v.norm()
        # sqrt(1 + 4 + 4) = sqrt(9) = 3
        assert abs(norm.value - 3.0) < 1e-10

    def test_norm_negative_components(self):
        """Test norm with negative components."""
        v = Vector(-3, -4)
        norm = v.norm()
        assert abs(norm.value - 5.0) < 1e-10


class TestVectorUnit:
    """Test unit vector (normalization)."""

    def test_unit_of_unit_vector(self):
        """Test unit vector of already unit vector."""
        v = Vector(1, 0)
        unit = v.unit()
        assert isinstance(unit, Vector)
        assert abs(unit[0].value - 1.0) < 1e-10
        assert abs(unit[1].value) < 1e-10

    def test_unit_vector_2d(self):
        """Test unit vector in 2D."""
        v = Vector(3, 4)
        unit = v.unit()
        assert isinstance(unit, Vector)
        # Unit should have norm 1
        norm = unit.norm()
        assert abs(norm.value - 1.0) < 1e-10

    def test_unit_vector_preserves_direction(self):
        """Test that unit vector preserves direction."""
        v = Vector(2, 0)
        unit = v.unit()
        # Should point in same direction
        assert unit[0].value > 0
        assert abs(unit[1].value) < 1e-10

    def test_unit_of_zero_vector_raises_error(self):
        """Test that unit vector of zero vector raises error."""
        v = Vector(0, 0)
        with pytest.raises((ZeroDivisionError, ValueError)):
            v.unit()


class TestVectorDotProduct:
    """Test dot product calculations."""

    def test_dot_product_orthogonal_vectors(self):
        """Test dot product of orthogonal vectors."""
        v1 = Vector(1, 0)
        v2 = Vector(0, 1)
        dot = v1.dot(v2)
        assert isinstance(dot, Real)
        assert abs(dot.value) < 1e-10

    def test_dot_product_same_vector(self):
        """Test dot product of vector with itself."""
        v = Vector(3, 4)
        dot = v.dot(v)
        # Should be norm squared = 25
        assert abs(dot.value - 25.0) < 1e-10

    def test_dot_product_2d(self):
        """Test 2D dot product."""
        v1 = Vector(1, 2)
        v2 = Vector(3, 4)
        dot = v1.dot(v2)
        # 1*3 + 2*4 = 11
        assert abs(dot.value - 11.0) < 1e-10

    def test_dot_product_3d(self):
        """Test 3D dot product."""
        v1 = Vector(1, 2, 3)
        v2 = Vector(4, 5, 6)
        dot = v1.dot(v2)
        # 1*4 + 2*5 + 3*6 = 32
        assert abs(dot.value - 32.0) < 1e-10


class TestVectorCrossProduct:
    """Test cross product (3D vectors only)."""

    def test_cross_product_3d(self):
        """Test cross product of 3D vectors."""
        v1 = Vector(1, 0, 0)
        v2 = Vector(0, 1, 0)
        cross = v1.cross(v2)
        assert isinstance(cross, Vector)
        assert len(cross) == 3
        # i x j = k = (0, 0, 1)
        assert abs(cross[0].value) < 1e-10
        assert abs(cross[1].value) < 1e-10
        assert abs(cross[2].value - 1.0) < 1e-10

    def test_cross_product_anticommutative(self):
        """Test that cross product is anticommutative."""
        v1 = Vector(1, 2, 3)
        v2 = Vector(4, 5, 6)
        cross1 = v1.cross(v2)
        cross2 = v2.cross(v1)
        # v1 x v2 = -(v2 x v1)
        for i in range(3):
            assert abs(cross1[i].value + cross2[i].value) < 1e-10

    def test_cross_product_same_vector_is_zero(self):
        """Test cross product of vector with itself."""
        v = Vector(1, 2, 3)
        cross = v.cross(v)
        # v x v = 0
        for i in range(3):
            assert abs(cross[i].value) < 1e-10

    def test_cross_product_perpendicular(self):
        """Test that cross product is perpendicular to both vectors."""
        v1 = Vector(1, 0, 0)
        v2 = Vector(0, 1, 0)
        cross = v1.cross(v2)
        # cross . v1 should be 0
        dot1 = cross.dot(v1)
        dot2 = cross.dot(v2)
        assert abs(dot1.value) < 1e-10
        assert abs(dot2.value) < 1e-10


class TestVectorEdgeCases:
    """Test edge cases and special vectors."""

    def test_zero_vector(self):
        """Test zero vector."""
        v = Vector(0, 0, 0)
        assert len(v) == 3
        assert v.norm().value == 0.0

    def test_very_large_components(self):
        """Test vector with very large components."""
        v = Vector(1e10, 2e10)
        assert v[0].value == 1e10

    def test_very_small_components(self):
        """Test vector with very small components."""
        v = Vector(1e-10, 2e-10)
        assert v[0].value == 1e-10

    def test_high_dimensional_vector(self):
        """Test vector in high dimensions."""
        coords = list(range(10))
        v = Vector(*coords)
        assert len(v) == 10

    def test_is_instance_of_mathvalue(self):
        """Test that Vector is instance of MathValue."""
        v = Vector(1, 2)
        assert isinstance(v, MathValue)


class TestVectorIntegration:
    """Integration tests for Vector class."""

    def test_workflow_arithmetic_chain(self):
        """Test workflow: vector arithmetic chain."""
        v1 = Vector(1, 2)
        v2 = Vector(3, 4)
        result = v1 + v2 * 2
        # (1, 2) + (6, 8) = (7, 10)
        assert result[0].value == 7.0
        assert result[1].value == 10.0

    def test_workflow_dot_product_norm(self):
        """Test workflow: dot product and norm."""
        v = Vector(3, 4)
        dot_self = v.dot(v)
        norm = v.norm()
        # norm^2 should equal dot product with self
        assert abs(dot_self.value - norm.value**2) < 1e-10

    def test_workflow_unit_normalization(self):
        """Test workflow: create unit vector."""
        v = Vector(5, 12)
        # norm should be 13 (5-12-13 triangle)
        norm = v.norm()
        assert abs(norm.value - 13.0) < 1e-10

        unit = v.unit()
        unit_norm = unit.norm()
        assert abs(unit_norm.value - 1.0) < 1e-10

    def test_workflow_cross_and_dot_products(self):
        """Test workflow: cross and dot products."""
        v1 = Vector(1, 0, 0)
        v2 = Vector(0, 1, 0)

        cross = v1.cross(v2)
        dot_result = v1.dot(v2)

        # Orthogonal vectors
        assert abs(dot_result.value) < 1e-10
        # Cross product perpendicular to both
        assert abs(cross.dot(v1).value) < 1e-10

    def test_workflow_vector_to_point(self):
        """Test workflow: using vector from origin to create point."""
        v = Vector(3, 4)
        coords = v.to_python()
        p = Point(*coords)
        assert len(p) == 2
        assert p[0].value == 3.0


