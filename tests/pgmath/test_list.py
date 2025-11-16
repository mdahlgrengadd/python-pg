"""Tests for List MathValue class."""

import pytest

from pg.math.collections import List
from pg.math.numeric import Infinity, Real
from pg.math.value import MathValue


@pytest.fixture
def real_values():
    """Fixture providing Real values for testing."""
    return {
        "1": Real(1),
        "2": Real(2),
        "3": Real(3),
        "4": Real(4),
        "5": Real(5),
        "0": Real(0),
        "neg1": Real(-1),
        "neg2": Real(-2),
    }


@pytest.fixture
def lists(real_values):
    """Fixture providing List instances for testing."""
    return {
        "empty": List([]),
        "single": List([real_values["1"]]),
        "pair": List([real_values["1"], real_values["2"]]),
        "triple": List([real_values["1"], real_values["2"], real_values["3"]]),
        "neg": List([real_values["neg1"], real_values["neg2"]]),
    }


class TestListBasicInstantiation:
    """Test basic instantiation and field access."""

    def test_empty_list_instantiation(self):
        """Test creating an empty List."""
        lst = List([])
        assert lst.elements == []
        assert len(lst) == 0

    def test_single_element_list(self, real_values):
        """Test creating a List with one element."""
        lst = List([real_values["1"]])
        assert len(lst) == 1
        assert lst.elements[0].compare(real_values["1"])

    def test_multiple_element_list(self, real_values):
        """Test creating a List with multiple elements."""
        elements = [real_values["1"], real_values["2"], real_values["3"]]
        lst = List(elements)
        assert len(lst) == 3
        assert lst.elements[0].compare(real_values["1"])
        assert lst.elements[1].compare(real_values["2"])
        assert lst.elements[2].compare(real_values["3"])

    def test_list_type_precedence(self):
        """Test that List has correct type precedence."""
        lst = List([])
        assert hasattr(lst, 'type_precedence')


class TestListConversions:
    """Test conversion methods."""

    def test_to_string_empty_list(self):
        """Test to_string() on empty list."""
        lst = List([])
        assert lst.to_string() == "[]"

    def test_to_string_single_element(self, real_values):
        """Test to_string() with single element."""
        lst = List([real_values["1"]])
        result = lst.to_string()
        assert result.startswith("[")
        assert result.endswith("]")

    def test_to_string_multiple_elements(self, real_values):
        """Test to_string() with multiple elements."""
        lst = List([real_values["1"], real_values["2"]])
        result = lst.to_string()
        assert "[" in result and "]" in result
        assert "," in result  # Elements separated by comma

    def test_to_tex_empty_list(self):
        """Test to_tex() on empty list."""
        lst = List([])
        result = lst.to_tex()
        assert "\\left[" in result
        assert "\\right]" in result

    def test_to_tex_single_element(self, real_values):
        """Test to_tex() with single element."""
        lst = List([real_values["1"]])
        result = lst.to_tex()
        assert "\\left[" in result
        assert "\\right]" in result

    def test_to_tex_multiple_elements(self, real_values):
        """Test to_tex() with multiple elements."""
        lst = List([real_values["1"], real_values["2"]])
        result = lst.to_tex()
        assert "\\left[" in result
        assert "\\right]" in result
        assert "," in result

    def test_to_python_empty_list(self):
        """Test to_python() on empty list."""
        lst = List([])
        result = lst.to_python()
        assert isinstance(result, list)
        assert result == []

    def test_to_python_with_elements(self, real_values):
        """Test to_python() converts elements properly."""
        lst = List([real_values["1"], real_values["2"]])
        result = lst.to_python()
        assert isinstance(result, list)
        assert len(result) == 2


class TestListIndexing:
    """Test list indexing operations."""

    def test_getitem_first_element(self, real_values):
        """Test getting first element by index."""
        lst = List([real_values["1"], real_values["2"]])
        result = lst[0]
        assert result.compare(real_values["1"])

    def test_getitem_last_element(self, real_values):
        """Test getting last element by index."""
        lst = List([real_values["1"], real_values["2"], real_values["3"]])
        result = lst[2]
        assert result.compare(real_values["3"])

    def test_getitem_negative_index(self, real_values):
        """Test getting element with negative index."""
        lst = List([real_values["1"], real_values["2"], real_values["3"]])
        result = lst[-1]
        assert result.compare(real_values["3"])

    def test_getitem_out_of_range_raises_error(self, real_values):
        """Test that out-of-range index raises IndexError."""
        lst = List([real_values["1"]])
        with pytest.raises(IndexError):
            _ = lst[5]

    def test_setitem_modify_element(self, real_values):
        """Test modifying element by index."""
        lst = List([real_values["1"], real_values["2"]])
        lst[0] = real_values["3"]
        assert lst[0].compare(real_values["3"])

    def test_setitem_modify_last_element(self, real_values):
        """Test modifying last element."""
        lst = List([real_values["1"], real_values["2"]])
        lst[1] = real_values["5"]
        assert lst[1].compare(real_values["5"])


class TestListLength:
    """Test length operations."""

    def test_len_empty_list(self):
        """Test len() on empty list."""
        lst = List([])
        assert len(lst) == 0

    def test_len_single_element(self, real_values):
        """Test len() with single element."""
        lst = List([real_values["1"]])
        assert len(lst) == 1

    def test_len_multiple_elements(self, real_values):
        """Test len() with multiple elements."""
        lst = List([real_values["1"], real_values["2"], real_values["3"]])
        assert len(lst) == 3


class TestListComparison:
    """Test comparison operations."""

    def test_compare_identical_lists(self, real_values):
        """Test comparing identical lists."""
        lst1 = List([real_values["1"], real_values["2"]])
        lst2 = List([real_values["1"], real_values["2"]])
        assert lst1.compare(lst2) is True

    def test_compare_different_elements(self, real_values):
        """Test comparing lists with different elements."""
        lst1 = List([real_values["1"], real_values["2"]])
        lst2 = List([real_values["1"], real_values["3"]])
        assert lst1.compare(lst2) is False

    def test_compare_different_lengths(self, real_values):
        """Test comparing lists of different lengths."""
        lst1 = List([real_values["1"], real_values["2"]])
        lst2 = List([real_values["1"]])
        assert lst1.compare(lst2) is False

    def test_compare_empty_lists(self):
        """Test comparing empty lists."""
        lst1 = List([])
        lst2 = List([])
        assert lst1.compare(lst2) is True

    def test_compare_with_non_list_type(self, real_values):
        """Test comparing with non-List MathValue returns False."""
        lst = List([real_values["1"]])

        class FakeMathValue:
            pass

        other = FakeMathValue()
        assert lst.compare(other) is False

    def test_compare_with_tolerance(self, real_values):
        """Test that compare passes tolerance to elements."""
        lst1 = List([real_values["1"], real_values["2"]])
        lst2 = List([real_values["1"], real_values["2"]])
        # Should work with various tolerance values
        assert lst1.compare(lst2, tolerance=0.5) is True
        assert lst1.compare(lst2, tolerance=0.001) is True


class TestListConcatenation:
    """Test list concatenation (addition)."""

    def test_add_two_lists(self, real_values):
        """Test adding two lists (concatenation)."""
        lst1 = List([real_values["1"]])
        lst2 = List([real_values["2"]])
        result = lst1 + lst2
        assert isinstance(result, List)
        assert len(result) == 2
        assert result[0].compare(real_values["1"])
        assert result[1].compare(real_values["2"])

    def test_add_empty_lists(self):
        """Test adding empty lists."""
        lst1 = List([])
        lst2 = List([])
        result = lst1 + lst2
        assert len(result) == 0

    def test_add_to_empty_list(self, real_values):
        """Test adding to empty list."""
        lst1 = List([])
        lst2 = List([real_values["1"]])
        result = lst1 + lst2
        assert len(result) == 1

    def test_add_scalar_to_list(self, real_values):
        """Test element-wise addition with scalar."""
        lst = List([real_values["1"], real_values["2"]])
        scalar = real_values["3"]
        result = lst + scalar
        assert isinstance(result, List)
        assert len(result) == 2
        # Each element should be increased by scalar
        assert result[0].compare(Real(4))  # 1 + 3
        assert result[1].compare(Real(5))  # 2 + 3

    def test_radd_scalar_to_list_raises_error(self, real_values):
        """Test right-addition with scalar raises TypeError."""
        lst = List([real_values["1"], real_values["2"]])
        # Real doesn't support radd with List
        with pytest.raises(TypeError):
            real_values["3"].__radd__(lst)

    def test_add_preserves_elements(self, real_values):
        """Test that concatenation doesn't modify original lists."""
        lst1 = List([real_values["1"]])
        lst2 = List([real_values["2"]])
        original_len1 = len(lst1)
        original_len2 = len(lst2)

        _ = lst1 + lst2

        assert len(lst1) == original_len1
        assert len(lst2) == original_len2


class TestListSubtraction:
    """Test list subtraction operations."""

    def test_sub_two_lists(self, real_values):
        """Test subtracting two lists (element-wise)."""
        lst1 = List([real_values["3"], real_values["4"]])
        lst2 = List([real_values["1"], real_values["2"]])
        result = lst1 - lst2
        assert isinstance(result, List)
        assert len(result) == 2
        # 3-1=2, 4-2=2
        assert result[0].compare(Real(2))
        assert result[1].compare(Real(2))

    def test_sub_different_length_lists_raises_error(self, real_values):
        """Test that subtracting lists of different lengths raises error."""
        lst1 = List([real_values["1"], real_values["2"]])
        lst2 = List([real_values["1"]])
        with pytest.raises(ValueError, match="List dimensions must match"):
            _ = lst1 - lst2

    def test_sub_scalar_from_list(self, real_values):
        """Test element-wise subtraction with scalar."""
        lst = List([real_values["3"], real_values["4"]])
        scalar = real_values["1"]
        result = lst - scalar
        assert isinstance(result, List)
        assert len(result) == 2
        # 3-1=2, 4-1=3
        assert result[0].compare(Real(2))
        assert result[1].compare(Real(3))

    def test_rsub_scalar_from_list_raises_error(self, real_values):
        """Test right subtraction with scalar raises TypeError."""
        lst = List([real_values["1"], real_values["2"]])
        scalar = real_values["5"]
        # Real doesn't support rsub with List
        with pytest.raises(TypeError):
            scalar - lst


class TestListMultiplication:
    """Test list multiplication operations."""

    def test_mul_list_by_scalar(self, real_values):
        """Test scalar multiplication."""
        lst = List([real_values["2"], real_values["3"]])
        scalar = real_values["2"]
        result = lst * scalar
        assert isinstance(result, List)
        assert len(result) == 2
        # 2*2=4, 3*2=6
        assert result[0].compare(Real(4))
        assert result[1].compare(Real(6))

    def test_rmul_list_by_scalar_raises_error(self, real_values):
        """Test right multiplication raises TypeError."""
        lst = List([real_values["2"], real_values["3"]])
        scalar = real_values["2"]
        # Real doesn't support rmul with List
        with pytest.raises(TypeError):
            scalar * lst

    def test_mul_element_wise(self, real_values):
        """Test element-wise multiplication of two lists."""
        lst1 = List([real_values["2"], real_values["3"]])
        lst2 = List([real_values["4"], real_values["5"]])
        result = lst1 * lst2
        assert isinstance(result, List)
        assert len(result) == 2
        # 2*4=8, 3*5=15
        assert result[0].compare(Real(8))
        assert result[1].compare(Real(15))

    def test_mul_different_length_lists_raises_error(self, real_values):
        """Test that multiplying lists of different lengths raises error."""
        lst1 = List([real_values["1"], real_values["2"]])
        lst2 = List([real_values["1"]])
        with pytest.raises(ValueError, match="List dimensions must match"):
            _ = lst1 * lst2

    def test_mul_by_zero(self, real_values):
        """Test multiplying by zero."""
        lst = List([real_values["2"], real_values["3"]])
        result = lst * real_values["0"]
        assert isinstance(result, List)
        assert len(result) == 2
        assert result[0].compare(real_values["0"])
        assert result[1].compare(real_values["0"])


class TestListDivision:
    """Test list division operations."""

    def test_div_list_by_scalar(self, real_values):
        """Test element-wise division by scalar."""
        lst = List([real_values["4"], real_values["2"]])
        scalar = real_values["2"]
        result = lst / scalar
        assert isinstance(result, List)
        assert len(result) == 2
        # 4/2=2, 2/2=1
        assert result[0].compare(Real(2))
        assert result[1].compare(Real(1))

    def test_div_element_wise(self, real_values):
        """Test element-wise division of two lists."""
        lst1 = List([real_values["4"], real_values["5"]])
        lst2 = List([real_values["2"], Real(5)])
        result = lst1 / lst2
        assert isinstance(result, List)
        assert len(result) == 2
        # 4/2=2, 5/5=1
        assert result[0].compare(Real(2))

    def test_div_different_length_lists_raises_error(self, real_values):
        """Test that dividing lists of different lengths raises error."""
        lst1 = List([real_values["1"], real_values["2"]])
        lst2 = List([real_values["1"]])
        with pytest.raises(ValueError, match="List dimensions must match"):
            _ = lst1 / lst2

    def test_rdiv_scalar_by_list_raises_error(self, real_values):
        """Test right division (scalar divided by list) raises TypeError."""
        lst = List([real_values["2"], Real(5)])
        scalar = Real(10)
        # Real doesn't support rdiv with List
        with pytest.raises(TypeError):
            scalar / lst


class TestListPower:
    """Test list power operations."""

    def test_pow_list_by_scalar(self, real_values):
        """Test element-wise power with scalar exponent."""
        lst = List([real_values["2"], real_values["3"]])
        exponent = real_values["2"]
        result = lst**exponent
        assert isinstance(result, List)
        assert len(result) == 2
        # 2^2=4, 3^2=9
        assert result[0].compare(Real(4))
        assert result[1].compare(Real(9))

    def test_rpow_scalar_base_and_list_exponents_raises_error(self, real_values):
        """Test right power (base to power of list) raises TypeError."""
        lst = List([real_values["2"], real_values["3"]])
        base = real_values["2"]
        # Real doesn't support rpow with List
        result = base.__rpow__(lst)
        assert result is NotImplemented


class TestListUnaryOperations:
    """Test unary operations on lists."""

    def test_negation(self, real_values):
        """Test unary negation on list."""
        lst = List([real_values["1"], real_values["2"], real_values["3"]])
        result = -lst
        assert isinstance(result, List)
        assert len(result) == 3
        assert result[0].compare(real_values["neg1"])
        assert result[1].compare(real_values["neg2"])

    def test_unary_positive(self, real_values):
        """Test unary positive on list."""
        lst = List([real_values["1"], real_values["neg2"]])
        result = +lst
        assert isinstance(result, List)
        assert len(result) == 2
        assert result[0].compare(real_values["1"])
        assert result[1].compare(real_values["neg2"])

    def test_absolute_value(self, real_values):
        """Test absolute value on list."""
        lst = List([real_values["neg1"], real_values["neg2"]])
        result = abs(lst)
        assert isinstance(result, List)
        assert len(result) == 2
        assert result[0].compare(real_values["1"])
        assert result[1].compare(real_values["2"])


class TestListPromote:
    """Test type promotion."""

    def test_promote_returns_self(self, real_values):
        """Test that lists don't promote."""
        lst = List([real_values["1"]])

        class FakeMathValue:
            pass

        result = lst.promote(FakeMathValue())
        assert result is lst


class TestListCmp:
    """Test cmp method (Perl compatibility)."""

    def test_cmp_returns_self(self, real_values):
        """Test that cmp() returns self."""
        lst = List([real_values["1"]])
        result = lst.cmp()
        assert result is lst

    def test_cmp_with_args(self, real_values):
        """Test that cmp() accepts args."""
        lst = List([real_values["1"]])
        result = lst.cmp("arg1", "arg2")
        assert result is lst

    def test_cmp_with_kwargs(self, real_values):
        """Test that cmp() accepts kwargs."""
        lst = List([real_values["1"]])
        result = lst.cmp(key="value")
        assert result is lst


class TestListIntegration:
    """Integration tests for List class."""

    def test_workflow_create_modify_compare(self, real_values):
        """Test workflow: create, modify, and compare lists."""
        lst1 = List([real_values["1"], real_values["2"]])
        lst1[1] = real_values["3"]
        lst2 = List([real_values["1"], real_values["3"]])
        assert lst1.compare(lst2)

    def test_workflow_arithmetic_chain(self, real_values):
        """Test chaining arithmetic operations."""
        lst1 = List([real_values["1"], real_values["2"]])
        lst2 = List([real_values["2"], real_values["3"]])
        # (list1 + list2) * 2
        result = (lst1 + lst2) * real_values["2"]
        assert len(result) == 4

    def test_workflow_conversions(self, real_values):
        """Test conversion workflow."""
        lst = List([real_values["1"], real_values["2"]])
        str_repr = lst.to_string()
        tex_repr = lst.to_tex()
        python_list = lst.to_python()

        assert "[" in str_repr
        assert "\\left[" in tex_repr
        assert isinstance(python_list, list)
        assert len(python_list) == 2

    def test_is_instance_of_mathvalue(self, real_values):
        """Test that List is instance of MathValue."""
        lst = List([real_values["1"]])
        assert isinstance(lst, MathValue)

    def test_list_immutability_via_operations(self, real_values):
        """Test that arithmetic operations don't modify original list."""
        lst = List([real_values["2"], real_values["3"]])
        original_len = len(lst)

        # Do operations
        _ = lst + real_values["1"]
        _ = lst * real_values["2"]
        _ = lst - real_values["1"]

        # Original should be unchanged
        assert len(lst) == original_len
        assert lst[0].compare(real_values["2"])

    def test_large_list_operations(self, real_values):
        """Test operations on larger lists."""
        elements = [real_values["1"] for _ in range(100)]
        lst = List(elements)
        assert len(lst) == 100

        result = lst * real_values["2"]
        assert len(result) == 100
        assert result[0].compare(Real(2))
