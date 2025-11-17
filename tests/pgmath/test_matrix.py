"""Tests for Matrix MathValue class."""

import numpy as np
import pytest

from pg.math.geometric import Matrix, Vector
from pg.math.numeric import Real
from pg.math.value import MathValue

class TestMatrixBasicInstantiation:
    """Test matrix construction scenarios."""

    def test_create_square_matrix(self):
        """Test creating a 2x2 matrix with numeric literals."""
        matrix = Matrix([[1, 2], [3, 4]])
        assert matrix.shape == (2, 2)
        assert matrix[0][0].value == 1.0
        assert matrix[1][1].value == 4.0

    def test_create_rectangular_matrix(self):
        """Test creating a rectangular (non-square) matrix."""
        matrix = Matrix([[1, 2, 3], [4, 5, 6]])
        assert matrix.shape == (2, 3)
        assert matrix[0][2].value == 3.0
        assert matrix[1][0].value == 4.0

    def test_create_matrix_from_real_objects(self):
        """Test creating a matrix from existing Real objects."""
        r11 = Real(2)
        r12 = Real(3)
        matrix = Matrix([[r11, r12], [Real(5), Real(7)]])
        assert matrix.shape == (2, 2)
        assert matrix.rows[0][0] is r11
        assert matrix.rows[0][1] is r12

    def test_create_empty_matrix(self):
        """Test creating an empty matrix."""
        matrix = Matrix([])
        assert matrix.shape == (0, 0)
        assert matrix.rows == []

    def test_create_single_element_matrix(self):
        """Test creating a 1x1 matrix."""
        matrix = Matrix([[42]])
        assert matrix.shape == (1, 1)
        assert matrix[0][0].value == 42.0

    def test_invalid_ragged_rows_raise_value_error(self):
        """Test that ragged row input raises ValueError."""
        with pytest.raises(ValueError):
            Matrix([[1, 2], [3]])

    def test_matrix_stores_mathvalue_instances(self):
        """Test that matrix entries are MathValue instances."""
        matrix = Matrix([[1, 2], [3, 4]])
        for row in matrix.rows:
            for element in row:
                assert isinstance(element, MathValue)

class TestMatrixDimensionAccess:
    """Test dimension helpers and indexing."""

    def test_shape_property_returns_rows_and_cols(self):
        """Test that shape reports (rows, cols)."""
        matrix = Matrix([[1, 2], [3, 4], [5, 6]])
        assert matrix.shape == (3, 2)

    def test_getitem_row_returns_list_of_mathvalues(self):
        """Test indexing by row returns list of MathValues."""
        matrix = Matrix([[1, 2, 3], [4, 5, 6]])
        second_row = matrix[1]
        assert isinstance(second_row, list)
        assert [cell.value for cell in second_row] == [4.0, 5.0, 6.0]

    def test_getitem_tuple_returns_single_element(self):
        """Test tuple indexing returns specific element."""
        matrix = Matrix([[1, 2], [3, 4]])
        element = matrix[1, 0]
        assert isinstance(element, MathValue)
        assert element.value == 3.0

    def test_column_returns_column_matrix(self):
        """Test extracting a column using 1-based indexing."""
        matrix = Matrix([[1, 2], [3, 4], [5, 6]])
        column = matrix.column(2)
        assert column.shape == (3, 1)
        assert [row[0].value for row in column.rows] == [2.0, 4.0, 6.0]

    def test_row_returns_row_matrix(self):
        """Test extracting a row using 1-based indexing."""
        matrix = Matrix([[1, 2, 3], [4, 5, 6]])
        row = matrix.row(2)
        assert row.shape == (1, 3)
        assert [cell.value for cell in row.rows[0]] == [4.0, 5.0, 6.0]

    def test_column_respects_one_based_indexing(self):
        """Test column helper uses Perl-style 1-based indices."""
        matrix = Matrix([[9, 8], [7, 6]])
        first_column = matrix.column(1)
        assert [row[0].value for row in first_column.rows] == [9.0, 7.0]

class TestMatrixConversions:
    """Test conversion helpers for Matrix."""

    def test_to_string_includes_brackets_and_values(self):
        """Test that to_string produces readable representation."""
        matrix = Matrix([[1, 2], [3, 4]])
        result = matrix.to_string()
        assert result.startswith('[')
        assert '1' in result and '4' in result

    def test_to_tex_uses_pmatrix_format(self):
        """Test LaTeX conversion uses pmatrix environment."""
        matrix = Matrix([[1, 2], [3, 4]])
        tex = matrix.to_tex()
        assert "\\begin{pmatrix}" in tex
        assert "\\end{pmatrix}" in tex
        assert "&" in tex

    def test_to_python_returns_nested_lists_of_floats(self):
        """Test that to_python emits nested Python lists."""
        matrix = Matrix([[Real(1), Real(2)], [Real(3), Real(4)]])
        assert matrix.to_python() == [[1.0, 2.0], [3.0, 4.0]]

    def test_to_numpy_returns_numpy_array(self):
        """Test conversion to numpy array preserves values."""
        matrix = Matrix([[1, 2], [3, 4]])
        np_array = matrix.to_numpy()
        assert isinstance(np_array, np.ndarray)
        assert np.array_equal(np_array, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_copy_creates_independent_instance(self):
        """Test copy() creates deep copy independent from original."""
        matrix = Matrix([[1, 2], [3, 4]])
        matrix_copy = matrix.copy()
        assert matrix_copy is not matrix
        matrix.rows[0][0] = Real(99)
        assert matrix_copy[0][0].value == 1.0

    def test_cmp_returns_self_for_compatibility(self):
        """Test cmp() returns the matrix itself for chaining."""
        matrix = Matrix([[1, 2], [3, 4]])
        assert matrix.cmp() is matrix

class TestMatrixArithmeticOperations:
    """Test arithmetic operators supported by Matrix."""

    def test_matrix_addition(self):
        """Test element-wise addition between matrices."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        result = m1 + m2
        assert result.to_python() == [[6.0, 8.0], [10.0, 12.0]]

    def test_matrix_subtraction(self):
        """Test element-wise subtraction between matrices."""
        m1 = Matrix([[5, 6], [7, 8]])
        m2 = Matrix([[1, 2], [3, 4]])
        result = m1 - m2
        assert result.to_python() == [[4.0, 4.0], [4.0, 4.0]]

    def test_matrix_addition_dimension_mismatch(self):
        """Test that mismatched shapes raise ValueError when adding."""
        m1 = Matrix([[1, 2]])
        m2 = Matrix([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            _ = m1 + m2

    def test_scalar_multiplication_right_operator(self):
        """Test multiplying matrix by scalar on the right."""
        matrix = Matrix([[1, 2], [3, 4]])
        result = matrix * 2
        assert result.to_python() == [[2.0, 4.0], [6.0, 8.0]]

    def test_scalar_multiplication_left_operator(self):
        """Test multiplying matrix by scalar on the left."""
        matrix = Matrix([[1, 2], [3, 4]])
        result = 3 * matrix
        assert result.to_python() == [[3.0, 6.0], [9.0, 12.0]]

    def test_scalar_division(self):
        """Test dividing matrix by scalar."""
        matrix = Matrix([[2, 4], [6, 8]])
        result = matrix / 2
        assert result.to_python() == [[1.0, 2.0], [3.0, 4.0]]

    def test_scalar_division_by_zero_raises(self):
        """Test dividing matrix by zero raises ZeroDivisionError."""
        matrix = Matrix([[1, 2]])
        with pytest.raises(ZeroDivisionError):
            _ = matrix / 0

    def test_matrix_multiplication_square(self):
        """Test multiplying two compatible matrices."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[2, 0], [1, 2]])
        result = m1 * m2
        assert result.to_python() == [[4.0, 4.0], [10.0, 8.0]]

    def test_matrix_multiplication_dimension_mismatch(self):
        """Test multiplying matrices with incompatible shapes raises ValueError."""
        m1 = Matrix([[1, 2]])
        m2 = Matrix([[1, 2]])
        with pytest.raises(ValueError):
            _ = m1 * m2

    def test_matrix_multiplied_by_vector_returns_vector(self):
        """Test multiplying matrix by vector returns Vector with expected components."""
        matrix = Matrix([[2, 0], [0, 1]])
        vector = Vector(1, 3)
        result = matrix * vector
        assert isinstance(result, Vector)
        assert result[0].value == 2.0
        assert result[1].value == 3.0

    def test_matrix_vector_dimension_mismatch(self):
        """Test multiplying by vector with mismatched dimension raises ValueError."""
        matrix = Matrix([[1, 2], [3, 4]])
        vector = Vector(1, 2, 3)
        with pytest.raises(ValueError):
            _ = matrix * vector

    def test_negation_negates_each_entry(self):
        """Test unary negation negates each element."""
        matrix = Matrix([[1, -2], [-3, 4]])
        negated = -matrix
        assert negated.to_python() == [[-1.0, 2.0], [3.0, -4.0]]

    def test_unary_positive_returns_same_values(self):
        """Test unary positive returns Matrix with same values."""
        matrix = Matrix([[1, -2], [3, -4]])
        positive = +matrix
        assert positive is not matrix
        assert positive.to_python() == [[1.0, -2.0], [3.0, -4.0]]

    def test_matrix_addition_with_scalar_raises_type_error(self):
        """Test adding scalar to matrix raises TypeError."""
        matrix = Matrix([[1, 2]])
        with pytest.raises(TypeError):
            _ = matrix + 1

class TestMatrixAdvancedOperations:
    """Test higher-level matrix operations."""

    def test_transpose_swaps_rows_and_columns(self):
        """Test transpose returns swapped dimensions."""
        matrix = Matrix([[1, 2, 3], [4, 5, 6]])
        transpose = matrix.transpose
        assert transpose.shape == (3, 2)
        assert transpose.to_python() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    def test_transpose_of_empty_matrix_is_empty(self):
        """Test transpose on empty matrix returns empty matrix."""
        matrix = Matrix([])
        transpose = matrix.transpose
        assert transpose.shape == (0, 0)
        assert transpose.rows == []

    def test_determinant_of_2x2_matrix(self):
        """Test determinant computation for simple matrix."""
        matrix = Matrix([[1, 2], [3, 4]])
        determinant = matrix.determinant()
        assert isinstance(determinant, Real)
        assert determinant.value == pytest.approx(-2.0)

    def test_determinant_non_square_raises_value_error(self):
        """Test determinant raises when matrix not square."""
        matrix = Matrix([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            _ = matrix.determinant()

    def test_inverse_of_simple_matrix(self):
        """Test inverse returns expected values for invertible matrix."""
        matrix = Matrix([[4, 7], [2, 6]])
        inverse = matrix.inverse()
        expected = [[0.6, -0.7], [-0.2, 0.4]]
        for r_idx, row in enumerate(inverse.to_python()):
            for c_idx, value in enumerate(row):
                assert value == pytest.approx(expected[r_idx][c_idx], rel=1e-7)

    def test_inverse_non_square_raises_value_error(self):
        """Test inverse raises ValueError for non-square matrix."""
        matrix = Matrix([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            _ = matrix.inverse()

    def test_inverse_singular_matrix_raises_value_error(self):
        """Test inverse raises when matrix is singular."""
        matrix = Matrix([[1, 2], [2, 4]])
        with pytest.raises(ValueError):
            _ = matrix.inverse()

    def test_trace_of_square_matrix(self):
        """Test trace sums diagonal for square matrix."""
        matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        trace = matrix.trace()
        assert isinstance(trace, Real)
        assert trace.value == pytest.approx(15.0)

    def test_trace_non_square_raises_value_error(self):
        """Test trace raises ValueError for non-square matrix."""
        matrix = Matrix([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            _ = matrix.trace()

class TestMatrixPowerOperations:
    """Test exponentiation helpers for matrices."""

    def test_power_zero_returns_identity_matrix(self):
        """Test raising matrix to zero yields identity matrix."""
        matrix = Matrix([[2, 0], [0, 3]])
        identity = matrix ** 0
        assert identity.to_python() == [[1.0, 0.0], [0.0, 1.0]]

    def test_power_positive_exponent(self):
        """Test positive integer powers multiply matrix repeatedly."""
        matrix = Matrix([[1, 1], [0, 1]])
        powered = matrix ** 3
        assert powered.to_python() == [[1.0, 3.0], [0.0, 1.0]]

    def test_power_negative_exponent_returns_inverse(self):
        """Test negative exponent returns matrix inverse."""
        matrix = Matrix([[4, 0], [0, 5]])
        power = matrix ** -1
        expected = matrix.inverse().to_python()
        for r_idx, row in enumerate(power.to_python()):
            for c_idx, value in enumerate(row):
                assert value == pytest.approx(expected[r_idx][c_idx])

    def test_power_non_square_matrix_raises_value_error(self):
        """Test exponentiation raises ValueError for non-square matrix."""
        matrix = Matrix([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            _ = matrix ** 2

    def test_power_with_non_integer_exponent_raises_type_error(self):
        """Test exponentiation with non-integer raises TypeError."""
        matrix = Matrix([[1, 0], [0, 1]])
        with pytest.raises(TypeError):
            _ = matrix ** 1.5

    def test_right_power_not_supported(self):
        """Test scalar ** matrix raises TypeError via __rpow__."""
        matrix = Matrix([[1, 0], [0, 1]])
        with pytest.raises(TypeError):
            _ = 2 ** matrix


class TestMatrixComparisonAndIntegration:
    """Test compare() and integration workflows."""

    def test_compare_identical_matrices_returns_true(self):
        """Test compare() returns True for identical matrices."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2], [3, 4]])
        assert m1.compare(m2) is True

    def test_compare_different_shapes_returns_false(self):
        """Test compare() returns False for different shapes."""
        m1 = Matrix([[1, 2]])
        m2 = Matrix([[1, 2], [3, 4]])
        assert m1.compare(m2) is False

    def test_compare_with_tolerance_accepts_small_differences(self):
        """Test compare() honors tolerance parameter."""
        m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
        m2 = Matrix([[1.0005, 2.0005], [2.9995, 4.0004]])
        assert m1.compare(m2, tolerance=0.001) is True

    def test_compare_detects_difference_without_tolerance(self):
        """Test compare() returns False when outside tolerance."""
        m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
        m2 = Matrix([[1.1, 2.0], [3.0, 4.0]])
        assert m1.compare(m2) is False

    def test_matrix_times_inverse_yields_identity(self):
        """Integration test: matrix multiplied by inverse yields identity."""
        matrix = Matrix([[3, 1], [0, 2]])
        identity = matrix * matrix.inverse()
        target = Matrix([[1, 0], [0, 1]])
        assert identity.compare(target, tolerance=1e-9) is True

    def test_matrix_transform_and_reverse_workflow(self):
        """Integration test: transform vector and recover original via inverse."""
        matrix = Matrix([[2, 0], [1, 1]])
        vector = Vector(3, 4)
        transformed = matrix * vector
        restored = matrix.inverse() * transformed
        assert abs(restored[0].value - 3.0) < 1e-9
        assert abs(restored[1].value - 4.0) < 1e-9

    def test_compare_with_non_matrix_returns_false(self):
        """Test compare() returns False when comparing non-matrix type."""
        matrix = Matrix([[1, 2], [3, 4]])
        assert matrix.compare(Vector(1, 2)) is False

class TestMatrixErrorHandling:
    """Test error paths for invalid matrix operations."""

    def test_column_index_out_of_range_raises_index_error(self):
        """Test requesting column beyond bounds raises IndexError."""
        matrix = Matrix([[1, 2], [3, 4]])
        with pytest.raises(IndexError):
            _ = matrix.column(3)

    def test_column_on_empty_matrix_raises_index_error(self):
        """Test requesting column on empty matrix raises IndexError."""
        matrix = Matrix([])
        with pytest.raises(IndexError):
            _ = matrix.column(1)

    def test_row_index_out_of_range_raises_index_error(self):
        """Test requesting invalid row raises IndexError."""
        matrix = Matrix([[1, 2]])
        with pytest.raises(IndexError):
            _ = matrix.row(3)

    def test_abs_not_supported_for_matrix(self):
        """Test abs(matrix) raises TypeError."""
        matrix = Matrix([[1, 2], [3, 4]])
        with pytest.raises(TypeError):
            _ = abs(matrix)

    def test_right_division_not_supported(self):
        """Test scalar divided by matrix raises TypeError."""
        matrix = Matrix([[1, 2], [3, 4]])
        with pytest.raises(TypeError):
            _ = 1 / matrix

    def test_matrix_multiplication_with_invalid_type_raises_type_error(self):
        """Test multiplying matrix by unsupported type raises TypeError."""
        matrix = Matrix([[1, 2], [3, 4]])
        with pytest.raises(TypeError):
            _ = matrix * "not a matrix"

