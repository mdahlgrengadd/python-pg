"""
Context system for mathematical expression parsing.

Contexts define the mathematical environment including:
- Allowed variables
- Constants and their values
- Available functions
- Operator precedence and associativity
- Reduction rules
- Tolerance settings for fuzzy comparison

Reference: lib/Parser/Context.pm and lib/Parser/Context/Default.pm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class Associativity(Enum):
    """Operator associativity."""

    LEFT = "left"
    RIGHT = "right"
    NONE = "none"


@dataclass
class OperatorConfig:
    """Configuration for an operator."""

    symbol: str
    precedence: int
    associativity: Associativity
    binary: bool = True  # True for binary, False for unary


@dataclass
class FunctionConfig:
    """Configuration for a function."""

    name: str
    min_args: int = 1
    max_args: int | None = None  # None means unlimited
    evaluator: Any = None  # Callable for evaluation (set at runtime)


@dataclass
class VariableConfig:
    """Configuration for a variable."""

    name: str
    latex: str | None = None  # LaTeX representation


@dataclass
class ToleranceConfig:
    """Tolerance settings for fuzzy comparison."""

    relative: float = 0.001
    absolute: float = 0.0001
    digits: int = 6  # Significant digits


@dataclass
class StringConfig:
    """Configuration for a string value in the context."""

    value: str
    alias: str | None = None
    case_sensitive: bool = False


class StringsManager:
    """Manager for string values in a context."""

    def __init__(self):
        self._strings: dict[str, StringConfig] = {}

    def add(self, **strings: dict) -> None:
        """
        Add strings to the context.

        Args:
            **strings: String names with optional configuration dicts
                      e.g., add(none={}, N={'alias': 'none'})
        """
        for name, config in strings.items():
            if config is None:
                config = {}

            alias = config.get('alias')
            case_sensitive = config.get('caseSensitive', False)

            self._strings[name] = StringConfig(
                value=name,
                alias=alias,
                case_sensitive=case_sensitive
            )

    def contains(self, value: str) -> bool:
        """Check if a string value is in the context."""
        for name, config in self._strings.items():
            if config.case_sensitive:
                if value == name or (config.alias and value == config.alias):
                    return True
            else:
                if value.lower() == name.lower() or (config.alias and value.lower() == config.alias.lower()):
                    return True
        return False

    def get_canonical(self, value: str) -> str | None:
        """Get the canonical form of a string value."""
        for name, config in self._strings.items():
            if config.case_sensitive:
                if value == name or (config.alias and value == config.alias):
                    return name
            else:
                if value.lower() == name.lower() or (config.alias and value.lower() == config.alias.lower()):
                    return name
        return None


@dataclass
class Context:
    """
    Mathematical context defining the parsing and evaluation environment.

    Attributes:
        name: Context name (e.g., "Numeric", "Complex", "Vector")
        variables: Allowed variables with their configurations
        constants: Mathematical constants and their values
        functions: Available functions and their configurations
        operators: Operator precedence and associativity
        tolerances: Fuzzy comparison tolerances
        flags: Additional context-specific flags
    """

    name: str
    variables: dict[str, VariableConfig] = field(default_factory=dict)
    constants: dict[str, float | complex] = field(default_factory=dict)
    functions: dict[str, FunctionConfig] = field(default_factory=dict)
    operators: dict[str, OperatorConfig] = field(default_factory=dict)
    tolerances: ToleranceConfig = field(default_factory=ToleranceConfig)
    flags: dict[str, Any] = field(default_factory=dict)
    strings: StringsManager = field(default_factory=StringsManager)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Context":
        """
        Load context from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Context instance
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Parse variables
        variables = {}
        for var_name in data.get("variables", []):
            if isinstance(var_name, dict):
                name = var_name["name"]
                latex = var_name.get("latex")
                variables[name] = VariableConfig(name=name, latex=latex)
            else:
                variables[var_name] = VariableConfig(name=var_name)

        # Parse constants
        constants = data.get("constants", {})

        # Parse functions
        functions = {}
        for func_data in data.get("functions", []):
            if isinstance(func_data, str):
                functions[func_data] = FunctionConfig(name=func_data)
            elif isinstance(func_data, dict):
                name = func_data["name"]
                functions[name] = FunctionConfig(
                    name=name,
                    min_args=func_data.get("min_args", 1),
                    max_args=func_data.get("max_args"),
                )

        # Parse operators
        operators = {}
        for op_data in data.get("operators", []):
            symbol = op_data["symbol"]
            operators[symbol] = OperatorConfig(
                symbol=symbol,
                precedence=op_data["precedence"],
                associativity=Associativity(op_data.get("associativity", "left")),
                binary=op_data.get("binary", True),
            )

        # Parse tolerances
        tol_data = data.get("tolerances", {})
        tolerances = ToleranceConfig(
            relative=tol_data.get("relative", 0.001),
            absolute=tol_data.get("absolute", 0.0001),
            digits=tol_data.get("digits", 6),
        )

        # Parse flags
        flags = data.get("flags", {})

        return cls(
            name=data["name"],
            variables=variables,
            constants=constants,
            functions=functions,
            operators=operators,
            tolerances=tolerances,
            flags=flags,
        )

    def get_flag(self, flag_name: str, default: Any = None) -> Any:
        """
        Get a context flag value.
        
        Args:
            flag_name: Name of the flag
            default: Default value if flag not set
        
        Returns:
            Flag value or default
        """
        return self.flags.get(flag_name, default)
    
    def set_flag(self, flag_name: str, value: Any) -> None:
        """
        Set a context flag.
        
        Args:
            flag_name: Name of the flag
            value: Value to set
        """
        self.flags[flag_name] = value
    
    def copy_flags_to(self, target_context: "Context") -> None:
        """
        Copy flags from this context to another.
        
        Args:
            target_context: Context to copy flags to
        """
        target_context.flags.update(self.flags)
    
    @classmethod
    def numeric(cls) -> "Context":
        """
        Create standard Numeric context.

        This is the most common context for basic mathematical expressions.
        """
        context = cls(name="Numeric")

        # Standard variables
        for var in ["x", "y", "z", "t", "r", "theta", "phi"]:
            context.variables[var] = VariableConfig(name=var)

        # Mathematical constants
        context.constants = {
            "pi": 3.141592653589793,
            "e": 2.718281828459045,
        }

        # Standard functions
        for func in [
            "sin",
            "cos",
            "tan",
            "sec",
            "csc",
            "cot",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            "ln",
            "log",
            "log10",
            "exp",
            "sqrt",
            "abs",
            "floor",
            "ceil",
            "round",
            "sign",
        ]:
            context.functions[func] = FunctionConfig(name=func, min_args=1, max_args=1)

        # Multi-argument functions
        context.functions["atan2"] = FunctionConfig(name="atan2", min_args=2, max_args=2)
        context.functions["max"] = FunctionConfig(name="max", min_args=1, max_args=None)
        context.functions["min"] = FunctionConfig(name="min", min_args=1, max_args=None)

        # Operators (following standard precedence)
        context.operators = {
            "||": OperatorConfig("||", precedence=1, associativity=Associativity.LEFT),
            "&&": OperatorConfig("&&", precedence=2, associativity=Associativity.LEFT),
            "<": OperatorConfig("<", precedence=3, associativity=Associativity.LEFT),
            "<=": OperatorConfig("<=", precedence=3, associativity=Associativity.LEFT),
            ">": OperatorConfig(">", precedence=3, associativity=Associativity.LEFT),
            ">=": OperatorConfig(">=", precedence=3, associativity=Associativity.LEFT),
            "==": OperatorConfig("==", precedence=3, associativity=Associativity.LEFT),
            "!=": OperatorConfig("!=", precedence=3, associativity=Associativity.LEFT),
            "+": OperatorConfig("+", precedence=4, associativity=Associativity.LEFT),
            "-": OperatorConfig("-", precedence=4, associativity=Associativity.LEFT),
            "*": OperatorConfig("*", precedence=5, associativity=Associativity.LEFT),
            "/": OperatorConfig("/", precedence=5, associativity=Associativity.LEFT),
            "%": OperatorConfig("%", precedence=5, associativity=Associativity.LEFT),
            "^": OperatorConfig("^", precedence=6, associativity=Associativity.RIGHT),
            "**": OperatorConfig("**", precedence=6, associativity=Associativity.RIGHT),
        }

        # Unary operators
        context.operators["-u"] = OperatorConfig(
            "-u", precedence=7, associativity=Associativity.RIGHT, binary=False
        )
        context.operators["+u"] = OperatorConfig(
            "+u", precedence=7, associativity=Associativity.RIGHT, binary=False
        )

        return context

    @classmethod
    def complex(cls) -> "Context":
        """
        Create Complex context for complex number expressions.
        """
        context = cls.numeric()
        context.name = "Complex"

        # Add imaginary unit
        context.constants["i"] = complex(0, 1)

        return context

    @classmethod
    def vector(cls) -> "Context":
        """
        Create Vector context for vector expressions.
        """
        context = cls.numeric()
        context.name = "Vector"

        # Add vector-specific functions
        context.functions["norm"] = FunctionConfig(name="norm", min_args=1, max_args=1)
        context.functions["unit"] = FunctionConfig(name="unit", min_args=1, max_args=1)
        context.functions["dot"] = FunctionConfig(name="dot", min_args=2, max_args=2)
        context.functions["cross"] = FunctionConfig(name="cross", min_args=2, max_args=2)

        return context

    @classmethod
    def interval(cls) -> "Context":
        """
        Create Interval context for interval notation.
        """
        context = cls.numeric()
        context.name = "Interval"

        # Add infinity
        context.constants["inf"] = float("inf")
        context.constants["infinity"] = float("inf")

        # Add interval-specific operators
        context.operators["U"] = OperatorConfig(
            "U", precedence=2, associativity=Associativity.LEFT
        )  # Union

        return context

    def get_operator_precedence(self, op: str, is_unary: bool = False) -> int:
        """
        Get the precedence of an operator.

        Args:
            op: Operator symbol
            is_unary: Whether this is a unary operator

        Returns:
            Precedence value (higher = binds tighter)
        """
        key = f"{op}u" if is_unary else op
        if key in self.operators:
            return self.operators[key].precedence
        return 0  # Unknown operator

    def get_operator_associativity(self, op: str, is_unary: bool = False) -> Associativity:
        """
        Get the associativity of an operator.

        Args:
            op: Operator symbol
            is_unary: Whether this is a unary operator

        Returns:
            Associativity
        """
        key = f"{op}u" if is_unary else op
        if key in self.operators:
            return self.operators[key].associativity
        return Associativity.LEFT  # Default

    def is_variable(self, name: str) -> bool:
        """Check if name is a valid variable in this context."""
        return name in self.variables

    def is_constant(self, name: str) -> bool:
        """Check if name is a constant in this context."""
        return name in self.constants

    def is_function(self, name: str) -> bool:
        """Check if name is a function in this context."""
        return name in self.functions

    def get_constant_value(self, name: str) -> float | complex:
        """Get the value of a constant."""
        return self.constants[name]
