"""Tests for Context system classes."""

import pytest

from pg.math.context import (
    Context,
    ContextFlags,
    ConstantManager,
    FunctionManager,
    OperatorManager,
    ParensManager,
    StringConfig,
    StringsManager,
    VariableManager,
    get_context,
    get_current_context,
)


class TestStringConfig:
    """Test StringConfig dataclass conversion."""

    def test_create_string_config(self):
        """Test creating a StringConfig."""
        config = StringConfig(value="test", alias="t", case_sensitive=True)
        assert config.value == "test"
        assert config.alias == "t"
        assert config.case_sensitive is True

    def test_string_config_defaults(self):
        """Test StringConfig with defaults."""
        config = StringConfig(value="test")
        assert config.value == "test"
        assert config.alias is None
        assert config.case_sensitive is False


class TestVariableManager:
    """Test VariableManager class."""

    def test_add_variable_positional(self):
        """Test adding variable with positional arguments."""
        mgr = VariableManager()
        mgr.add("x", "Real")
        var_info = mgr.get("x")
        assert var_info["type"] == "Real"
        assert var_info["options"] == {}

    def test_add_variable_keyword(self):
        """Test adding variable with keyword arguments."""
        mgr = VariableManager()
        mgr.add(k="Real")
        var_info = mgr.get("k")
        assert var_info["type"] == "Real"

    def test_set_variable_options(self):
        """Test setting variable options."""
        mgr = VariableManager()
        mgr.add("x", "Real")
        mgr.set("x", limits=[0, 10])
        var_info = mgr.get("x")
        assert var_info["options"]["limits"] == [0, 10]

    def test_remove_variable(self):
        """Test removing a variable."""
        mgr = VariableManager()
        mgr.add("x", "Real")
        mgr.remove("x")
        assert mgr.get("x") is None

    def test_copy_variable_manager(self):
        """Test copying a VariableManager."""
        mgr = VariableManager()
        mgr.add("x", "Real")
        mgr_copy = mgr.copy()
        assert mgr_copy.get("x")["type"] == "Real"
        mgr_copy.add("y", "Complex")
        assert mgr.get("y") is None  # Original unchanged


class TestConstantManager:
    """Test ConstantManager class."""

    def test_add_constant(self):
        """Test adding a constant."""
        mgr = ConstantManager()
        mgr.add("pi", 3.14159)
        assert mgr.get("pi") == 3.14159

    def test_constant_contains(self):
        """Test checking if constant exists."""
        mgr = ConstantManager()
        mgr.add("e", 2.718)
        assert "e" in mgr
        assert "pi" not in mgr

    def test_copy_constant_manager(self):
        """Test copying a ConstantManager."""
        mgr = ConstantManager()
        mgr.add("pi", 3.14159)
        mgr_copy = mgr.copy()
        assert mgr_copy.get("pi") == 3.14159


class TestFunctionManager:
    """Test FunctionManager class."""

    def test_add_function(self):
        """Test adding a function."""
        mgr = FunctionManager()
        mgr.add("sin")
        assert "sin" in mgr
        assert mgr.get("sin") == {}

    def test_function_options(self):
        """Test function with options."""
        mgr = FunctionManager()
        mgr.add("f", {"key": "value"})
        assert mgr.get("f")["key"] == "value"

    def test_enable_disable_functions(self):
        """Test enabling and disabling functions."""
        mgr = FunctionManager()
        mgr.add("sin")
        mgr.disable("sin")
        assert "sin" not in mgr
        mgr.enable("sin")
        assert "sin" in mgr


class TestOperatorManager:
    """Test OperatorManager class."""

    def test_add_operator(self):
        """Test adding an operator."""
        mgr = OperatorManager()
        mgr.add("+", priority=5)
        op_info = mgr.get("+")
        assert op_info["priority"] == 5

    def test_list_operators(self):
        """Test listing operators."""
        mgr = OperatorManager()
        mgr.add("+")
        mgr.add("-")
        ops = mgr.list()
        assert "+" in ops
        assert "-" in ops


class TestStringsManager:
    """Test StringsManager class."""

    def test_add_string(self):
        """Test adding a string."""
        mgr = StringsManager()
        mgr.add("none", {"alias": "N"})
        config = mgr.get("none")
        assert config.value == "none"
        assert config.alias == "N"

    def test_copy_strings_manager(self):
        """Test copying StringsManager."""
        mgr = StringsManager()
        mgr.add("test", {"alias": "t"})
        mgr_copy = mgr.copy()
        assert mgr_copy.get("test").alias == "t"


class TestContextFlags:
    """Test ContextFlags class."""

    def test_default_flags(self):
        """Test default flag values."""
        flags = ContextFlags()
        assert flags.get("tolerance") == 0.001
        assert flags.get("tolType") == "relative"

    def test_set_flags(self):
        """Test setting flags."""
        flags = ContextFlags()
        flags.set(tolerance=0.01, reduceConstants=False)
        assert flags.get("tolerance") == 0.01
        assert flags.get("reduceConstants") == 0

    def test_copy_flags(self):
        """Test copying flags."""
        flags = ContextFlags()
        flags.set(tolerance=0.01)
        flags_copy = flags.copy()
        assert flags_copy.get("tolerance") == 0.01


class TestParensManager:
    """Test ParensManager class."""

    def test_set_parens(self):
        """Test setting parenthesis configuration."""
        mgr = ParensManager()
        mgr.set("(", type="point")
        config = mgr.get("(")
        assert config["type"] == "point"

    def test_copy_parens_manager(self):
        """Test copying ParensManager."""
        mgr = ParensManager()
        mgr.set("(", type="point")
        mgr_copy = mgr.copy()
        assert mgr_copy.get("(")["type"] == "point"


class TestContext:
    """Test main Context class."""

    def test_create_numeric_context(self):
        """Test creating a Numeric context."""
        ctx = Context("Numeric")
        assert ctx.name == "Numeric"
        assert "x" in ctx.variables.list()
        assert "pi" in ctx.constants.list()

    def test_create_complex_context(self):
        """Test creating a Complex context."""
        ctx = Context("Complex")
        assert ctx.name == "Complex"
        assert "i" in ctx.constants.list()

    def test_context_copy(self):
        """Test copying a context."""
        # Use a non-singleton context name to avoid singleton issues
        ctx = Context("TestCopyContext")
        ctx.variables.add("t", "Real")
        ctx_copy = ctx.copy()
        assert ctx_copy.name == "TestCopyContext"
        assert "t" in ctx_copy.variables.list()
        ctx_copy.variables.add("y", "Real")
        assert "y" not in ctx.variables.list()  # Original unchanged
        assert "t" in ctx.variables.list()  # Original still has t

    def test_context_storage(self):
        """Test context storage (dict-like access)."""
        ctx = Context("Numeric")
        ctx["test"] = "value"
        assert ctx["test"] == "value"

    def test_get_context(self):
        """Test get_context function."""
        # Test that get_context can create a context
        # Use a unique name to avoid test interference
        ctx = get_context("TestGetContext")
        assert isinstance(ctx, Context)
        assert ctx.name == "TestGetContext"
        # Verify it has the expected structure
        assert ctx.variables is not None
        assert ctx.constants is not None

    def test_get_current_context(self):
        """Test get_current_context function."""
        ctx = get_current_context()
        assert isinstance(ctx, Context)

