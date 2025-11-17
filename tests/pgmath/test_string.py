"""Tests for String MathValue class."""

import pytest

from pg.math.collections import String
from pg.math.value import MathValue


class TestStringBasicInstantiation:
    """Test basic instantiation and field access."""

    def test_empty_string_instantiation(self):
        """Test creating a String with empty value."""
        s = String("")
        assert s.value == ""
        assert str(s) == ""

    def test_simple_string_instantiation(self):
        """Test creating a String with simple value."""
        s = String("hello")
        assert s.value == "hello"

    def test_string_with_spaces(self):
        """Test creating a String with spaces."""
        s = String("hello world")
        assert s.value == "hello world"

    def test_string_with_special_chars(self):
        """Test creating a String with special characters."""
        s = String("!@#$%^&*()")
        assert s.value == "!@#$%^&*()"

    def test_string_with_numbers(self):
        """Test creating a String with numeric characters."""
        s = String("12345")
        assert s.value == "12345"

    def test_string_with_latex(self):
        """Test creating a String with LaTeX content."""
        s = String(r"\alpha + \beta")
        assert s.value == r"\alpha + \beta"

    def test_string_with_newlines(self):
        """Test creating a String with newline characters."""
        s = String("line1\nline2")
        assert s.value == "line1\nline2"

    def test_string_type_precedence(self):
        """Test that String has correct type precedence."""
        s = String("test")
        assert hasattr(s, 'type_precedence')


class TestStringConversions:
    """Test conversion methods."""

    def test_to_string(self):
        """Test to_string() method."""
        s = String("hello")
        assert s.to_string() == "hello"

    def test_to_python(self):
        """Test to_python() method."""
        s = String("world")
        assert s.to_python() == "world"

    def test_to_tex_simple(self):
        """Test to_tex() method with simple string."""
        s = String("hello")
        assert s.to_tex() == "\\text{hello}"

    def test_to_tex_with_special_chars(self):
        """Test to_tex() method with special characters."""
        s = String("a&b")
        # to_tex wraps in \text{...}
        assert s.to_tex() == "\\text{a&b}"

    def test_to_tex_empty_string(self):
        """Test to_tex() method with empty string."""
        s = String("")
        assert s.to_tex() == "\\text{}"

    def test_to_tex_with_braces(self):
        """Test to_tex() method with braces in content."""
        s = String("{x}")
        assert s.to_tex() == "\\text{{x}}"


class TestStringLength:
    """Test length operations."""

    def test_len_empty_string(self):
        """Test len() on empty String."""
        s = String("")
        assert len(s) == 0

    def test_len_simple_string(self):
        """Test len() on simple String."""
        s = String("hello")
        assert len(s) == 5

    def test_len_string_with_spaces(self):
        """Test len() counts spaces."""
        s = String("hello world")
        assert len(s) == 11

    def test_len_unicode_string(self):
        """Test len() on unicode String."""
        s = String("café")
        assert len(s) == 4


class TestStringComparison:
    """Test comparison operations."""

    def test_compare_identical_strings(self):
        """Test comparing identical strings."""
        s1 = String("hello")
        s2 = String("hello")
        assert s1.compare(s2) is True

    def test_compare_different_strings(self):
        """Test comparing different strings."""
        s1 = String("hello")
        s2 = String("world")
        assert s1.compare(s2) is False

    def test_compare_empty_strings(self):
        """Test comparing empty strings."""
        s1 = String("")
        s2 = String("")
        assert s1.compare(s2) is True

    def test_compare_case_sensitive(self):
        """Test that comparison is case-sensitive."""
        s1 = String("Hello")
        s2 = String("hello")
        assert s1.compare(s2) is False

    def test_compare_whitespace_sensitive(self):
        """Test that comparison is whitespace-sensitive."""
        s1 = String("hello world")
        s2 = String("hello  world")
        assert s1.compare(s2) is False

    def test_compare_with_non_string_type(self):
        """Test comparing with non-String MathValue returns False."""
        s = String("5")
        # Create a simple object that acts like a MathValue
        class FakeMathValue:
            pass

        other = FakeMathValue()
        assert s.compare(other) is False

    def test_compare_ignores_tolerance_parameter(self):
        """Test that tolerance parameter is accepted but ignored for strings."""
        s1 = String("test")
        s2 = String("test")
        # String comparison should work same with any tolerance
        assert s1.compare(s2, tolerance=0.5) is True
        assert s1.compare(s2, tolerance=0.001) is True

    def test_compare_ignores_mode_parameter(self):
        """Test that mode parameter is accepted but ignored for strings."""
        s1 = String("test")
        s2 = String("test")
        # String comparison should work same with any mode
        assert s1.compare(s2, mode="relative") is True
        assert s1.compare(s2, mode="absolute") is True


class TestStringConcatenation:
    """Test string concatenation operations."""

    def test_add_two_strings(self):
        """Test adding two String objects."""
        s1 = String("hello")
        s2 = String(" world")
        result = s1 + s2
        assert isinstance(result, String)
        assert result.value == "hello world"

    def test_add_string_and_python_str(self):
        """Test adding String and Python str."""
        s = String("hello")
        result = s + " world"
        assert isinstance(result, String)
        assert result.value == "hello world"

    def test_radd_python_str_and_string(self):
        """Test right-adding Python str and String."""
        s = String("world")
        result = "hello " + s
        assert isinstance(result, String)
        assert result.value == "hello world"

    def test_add_empty_strings(self):
        """Test adding empty strings."""
        s1 = String("")
        s2 = String("")
        result = s1 + s2
        assert result.value == ""

    def test_add_to_empty_string(self):
        """Test adding to empty string."""
        s1 = String("")
        s2 = String("hello")
        result = s1 + s2
        assert result.value == "hello"

    def test_add_empty_string_to_text(self):
        """Test adding empty string to text."""
        s1 = String("hello")
        s2 = String("")
        result = s1 + s2
        assert result.value == "hello"

    def test_add_chain_multiple(self):
        """Test chaining multiple additions."""
        s1 = String("a")
        s2 = String("b")
        s3 = String("c")
        result = s1 + s2 + s3
        assert result.value == "abc"

    def test_add_with_invalid_type_returns_not_implemented(self):
        """Test adding with invalid type returns NotImplemented."""
        s = String("hello")
        result = s.__add__(123)
        assert result is NotImplemented

    def test_radd_with_invalid_type_returns_not_implemented(self):
        """Test right-adding with invalid type returns NotImplemented."""
        s = String("hello")
        result = s.__radd__(123)
        assert result is NotImplemented

    def test_add_python_str_prefix(self):
        """Test Python str prefix concatenation."""
        s = String("world")
        result = "hello " + s
        assert result.value == "hello world"

    def test_add_preserves_content(self):
        """Test that concatenation preserves special characters."""
        s1 = String("a\nb")
        s2 = String("c\nd")
        result = s1 + s2
        assert result.value == "a\nbc\nd"


class TestStringRepetition:
    """Test string repetition operations."""

    def test_mul_string_by_int(self):
        """Test multiplying String by integer."""
        s = String("ab")
        result = s * 3
        assert isinstance(result, String)
        assert result.value == "ababab"

    def test_rmul_int_by_string(self):
        """Test multiplying integer by String (rmul)."""
        s = String("x")
        result = 4 * s
        assert isinstance(result, String)
        assert result.value == "xxxx"

    def test_mul_by_zero(self):
        """Test multiplying by zero."""
        s = String("hello")
        result = s * 0
        assert result.value == ""

    def test_mul_by_one(self):
        """Test multiplying by one."""
        s = String("hello")
        result = s * 1
        assert result.value == "hello"

    def test_mul_empty_string(self):
        """Test multiplying empty string."""
        s = String("")
        result = s * 5
        assert result.value == ""

    def test_mul_with_invalid_type_returns_not_implemented(self):
        """Test multiplying by invalid type returns NotImplemented."""
        s = String("hello")
        result = s.__mul__("3")
        assert result is NotImplemented

    def test_rmul_with_invalid_type_returns_not_implemented(self):
        """Test right-multiplying by invalid type returns NotImplemented."""
        s = String("hello")
        result = s.__rmul__("3")
        assert result is NotImplemented

    def test_mul_preserves_content(self):
        """Test that repetition preserves special characters."""
        s = String("a\nb")
        result = s * 2
        assert result.value == "a\nba\nb"


class TestStringPromote:
    """Test type promotion."""

    def test_promote_returns_self(self):
        """Test that strings don't promote."""
        s = String("hello")
        # String.promote() always returns self
        result = s.promote(String("world"))
        assert result is s

    def test_promote_with_different_type(self):
        """Test promote with a different type still returns self."""
        s = String("hello")
        # Create a fake MathValue
        class FakeValue:
            pass
        result = s.promote(FakeValue())
        assert result is s


class TestStringCmp:
    """Test cmp method (Perl compatibility)."""

    def test_cmp_returns_self(self):
        """Test that cmp() returns self for compatibility."""
        s = String("hello")
        result = s.cmp()
        assert result is s

    def test_cmp_with_args(self):
        """Test that cmp() accepts args."""
        s = String("hello")
        result = s.cmp("arg1", "arg2")
        assert result is s

    def test_cmp_with_kwargs(self):
        """Test that cmp() accepts kwargs."""
        s = String("hello")
        result = s.cmp(key="value")
        assert result is s


class TestStringUnsupportedOperations:
    """Test that unsupported operations raise TypeError."""

    def test_subtraction_raises_error(self):
        """Test that subtraction raises TypeError."""
        s1 = String("hello")
        s2 = String("world")
        with pytest.raises(TypeError, match="String does not support subtraction"):
            s1 - s2

    def test_right_subtraction_raises_error(self):
        """Test that right subtraction raises TypeError."""
        s = String("world")
        with pytest.raises(TypeError, match="String does not support subtraction"):
            s.__rsub__("hello")

    def test_division_raises_error(self):
        """Test that division raises TypeError."""
        s = String("hello")
        with pytest.raises(TypeError, match="String does not support division"):
            s / 2

    def test_right_division_raises_error(self):
        """Test that right division raises TypeError."""
        s = String("hello")
        with pytest.raises(TypeError, match="String does not support division"):
            s.__rtruediv__(10)

    def test_power_raises_error(self):
        """Test that exponentiation raises TypeError."""
        s = String("hello")
        with pytest.raises(TypeError, match="String does not support exponentiation"):
            s ** 2

    def test_right_power_raises_error(self):
        """Test that right power raises TypeError."""
        s = String("hello")
        with pytest.raises(TypeError, match="String does not support exponentiation"):
            s.__rpow__(2)

    def test_negation_raises_error(self):
        """Test that negation raises TypeError."""
        s = String("hello")
        with pytest.raises(TypeError, match="String does not support negation"):
            -s

    def test_unary_positive_raises_error(self):
        """Test that unary positive raises TypeError."""
        s = String("hello")
        with pytest.raises(TypeError, match="String does not support unary positive"):
            +s

    def test_absolute_value_raises_error(self):
        """Test that absolute value raises TypeError."""
        s = String("hello")
        with pytest.raises(TypeError, match="String does not support absolute value"):
            abs(s)


class TestStringIntegration:
    """Integration tests for String class."""

    def test_workflow_concatenate_and_compare(self):
        """Test typical workflow: concatenate strings and compare."""
        s1 = String("hello")
        s2 = String(" ")
        s3 = String("world")
        result = s1 + s2 + s3
        expected = String("hello world")
        assert result.compare(expected) is True

    def test_workflow_repetition_and_length(self):
        """Test workflow: repeat string and check length."""
        s = String("ab")
        repeated = s * 3
        assert len(repeated) == 6
        assert repeated.value == "ababab"

    def test_workflow_mixed_operations(self):
        """Test workflow: mix different operations."""
        s1 = String("x")
        s2 = s1 * 3  # "xxx"
        s3 = String("y")
        s4 = s2 + s3  # "xxxy"
        assert len(s4) == 4
        assert s4.compare(String("xxxy")) is True

    def test_workflow_latex_conversion(self):
        """Test workflow: create and convert to LaTeX."""
        s = String("α + β")
        latex = s.to_tex()
        assert "\\text{" in latex
        assert "α + β" in latex

    def test_string_immutability_on_operations(self):
        """Test that operations don't modify original strings."""
        s1 = String("hello")
        s2 = String("world")
        original_s1 = s1.value
        original_s2 = s2.value

        # Do operations
        _ = s1 + s2
        _ = s1 * 2

        # Originals should be unchanged
        assert s1.value == original_s1
        assert s2.value == original_s2

    def test_is_instance_of_mathvalue(self):
        """Test that String is instance of MathValue."""
        s = String("test")
        assert isinstance(s, MathValue)

    def test_unicode_operations(self):
        """Test operations with unicode strings."""
        s1 = String("café")
        s2 = String(" français")
        result = s1 + s2
        assert result.value == "café français"
        # Unicode string length
        assert len(result) == 13
