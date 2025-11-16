"""parserFormulaWithUnits.py - Formula with units support.

Implements FormulaWithUnits class for formulas that have physical units attached.
This is a wrapper around Formula that stores units separately.

Based on macros/parsers/parserFormulaWithUnits.pl
"""

from typing import Any, Dict, Optional, Union
import re


class FormulaWithUnits:
    """A formula with physical units attached.
    
    Usage:
        FormulaWithUnits("3*x + 1 ft")  # Parse from string
        FormulaWithUnits(Formula("3*x + 1"), "ft")  # From Formula object
        
    The formula and units are stored separately and units are preserved
    through operations like differentiation and evaluation.
    """
    
    def __init__(self, formula_str: Union[str, Any], units: Optional[str] = None, **options):
        """Initialize a FormulaWithUnits object.
        
        Args:
            formula_str: Either "formula units" string or Formula object
            units: Unit string (if formula_str is a Formula)
            **options: Additional options (context, newUnit, etc.)
        """
        self.context = options.pop('context', None)
        
        # Get context if not provided
        if self.context is None:
            try:
                from pg.macros.core.mathobjects import Context
                self.context = Context()
            except Exception:
                self.context = None
        
        # Parse formula and units
        if units is not None:
            # Formula or string provided with separate units
            if isinstance(formula_str, str):
                # It's a string - parse it as Formula
                try:
                    from pg.math.formula import Formula
                    
                    # Get variables from context if available
                    variables = None
                    if self.context and hasattr(self.context, 'variables'):
                        variables = self.context.variables.list()
                    
                    self.formula = Formula(formula_str, variables=variables, context=self.context)
                except Exception as e:
                    # Fallback: keep as string
                    self.formula = formula_str
            else:
                # It's already a Formula object
                self.formula = formula_str
            self.units_str = str(units).strip()
        else:
            # String format: "formula units"
            formula_str = str(formula_str)
            self.formula, self.units_str = self._parse_formula_with_units(formula_str)
        
        # Get unit information from context if available
        self.units_data = self._get_unit_data(self.units_str)
        
    def _parse_formula_with_units(self, input_str: str) -> tuple[Any, str]:
        """Parse a string like '3*x + 1 ft' into formula and units.
        
        Args:
            input_str: String containing formula and units
            
        Returns:
            Tuple of (formula, units_string)
        """
        # Try to import Formula
        try:
            from pg.math.formula import Formula
        except ImportError:
            Formula = None
        
        # Strategy: Find the last "word" that could be a unit
        # Units are typically at the end: "3*x+1 ft" or "x^2 m/s"
        
        # Split by whitespace from the right
        parts = input_str.rsplit(None, 1)  # Split on last whitespace
        
        if len(parts) == 2:
            formula_part, unit_part = parts
            
            # Check if unit_part looks like a unit (not a variable/number)
            # Units are alphabetic, possibly with / like "m/s"
            if self._looks_like_unit(unit_part):
                # Parse the formula part
                if Formula:
                    try:
                        # Get variables from context if available
                        variables = None
                        if self.context and hasattr(self.context, 'variables'):
                            variables = self.context.variables.list()
                        
                        formula = Formula(formula_part, variables=variables, context=self.context)
                        return formula, unit_part
                    except Exception:
                        # If Formula creation fails, store as string
                        pass
                
                # Fallback: store as string
                return formula_part, unit_part
        
        # If we can't split, treat whole thing as formula with no units
        if Formula:
            try:
                return Formula(input_str, context=self.context), ''
            except Exception:
                pass
        
        return input_str, ''
    
    def _looks_like_unit(self, text: str) -> bool:
        """Check if text looks like a unit string.
        
        Args:
            text: String to check
            
        Returns:
            True if it looks like a unit
        """
        # Units are typically:
        # - Alphabetic: ft, m, s
        # - With divisions: m/s, ft/s
        # - With powers: m^2, ft^3
        # - Combinations: kg*m/s^2
        
        # Simple check: starts with letter, contains only letters, /, *, ^, digits
        return bool(re.match(r'^[a-zA-Z][a-zA-Z0-9/*^]*$', text))
    
    def _get_unit_data(self, units_str: str) -> Dict[str, Any]:
        """Get unit conversion data from context.
        
        Args:
            units_str: Unit string like 'ft' or 'm/s'
            
        Returns:
            Dictionary with unit information
        """
        # Try to get from Units context
        if hasattr(self.context, 'isUnit') and self.context.isUnit(units_str):
            # Get conversion factor if available
            if hasattr(self.context, 'constants'):
                const = self.context.constants.get(units_str)
                if const is not None:
                    return {'factor': float(const), 'name': units_str}
        
        # Default: unit with factor 1 (no conversion)
        return {'factor': 1.0, 'name': units_str}
    
    def D(self, var: str = 'x') -> 'FormulaWithUnits':
        """Compute derivative with respect to variable.
        
        When differentiating a formula with units, the units change.
        For example, d/dt of "position in ft" gives "velocity in ft/s"
        
        Args:
            var: Variable to differentiate with respect to
            
        Returns:
            New FormulaWithUnits with derivative
        """
        # Differentiate the formula
        if hasattr(self.formula, 'D'):
            deriv_formula = self.formula.D(var)
        else:
            # Fallback: can't differentiate
            raise ValueError(f"Cannot differentiate {type(self.formula)}")
        
        # Update units: add /var_unit to units
        # If t has units 's', then d/dt changes 'ft' to 'ft/s'
        var_unit = self._get_variable_unit(var)
        
        if var_unit:
            new_units = f"{self.units_str}/{var_unit}"
        else:
            new_units = self.units_str
        
        return FormulaWithUnits(deriv_formula, new_units, context=self.context)
    
    def _get_variable_unit(self, var: str) -> Optional[str]:
        """Get the unit assigned to a variable.
        
        Args:
            var: Variable name
            
        Returns:
            Unit string or None
        """
        if hasattr(self.context, 'getVariableUnit'):
            return self.context.getVariableUnit(var)
        return None
    
    def eval(self, **values) -> 'NumberWithUnits':
        """Evaluate formula at given variable values.
        
        Args:
            **values: Variable assignments (x=1, t=2, etc.)
            
        Returns:
            NumberWithUnits with the result
        """
        # Evaluate the formula
        if hasattr(self.formula, 'eval'):
            result = self.formula.eval(**values)
        else:
            # Try to evaluate as callable
            result = self.formula(**values) if callable(self.formula) else self.formula
        
        # Return as NumberWithUnits
        return NumberWithUnits(float(result), self.units_str, context=self.context)
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.formula} {self.units_str}"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"FormulaWithUnits('{self.formula}', '{self.units_str}')"


class NumberWithUnits:
    """A numeric value with physical units.
    
    Usage:
        NumberWithUnits("3.5 ft")  # Parse from string  
        NumberWithUnits(3.5, "ft")  # From number and unit
    """
    
    def __init__(self, value: Union[str, float], units: Optional[str] = None, **options):
        """Initialize a NumberWithUnits object.
        
        Args:
            value: Either "number units" string or numeric value
            units: Unit string (if value is numeric)
            **options: Additional options (context, etc.)
        """
        self.context = options.pop('context', None)
        
        # Parse value and units
        if units is not None:
            self.value = float(value)
            self.units_str = str(units).strip()
        else:
            # String format: "3.5 ft"
            value_str = str(value)
            self.value, self.units_str = self._parse_number_with_units(value_str)
        
        # Get unit information
        self.units_data = self._get_unit_data(self.units_str)
    
    def _parse_number_with_units(self, input_str: str) -> tuple[float, str]:
        """Parse a string like '3.5 ft' into number and units.
        
        Args:
            input_str: String containing number and units
            
        Returns:
            Tuple of (number, units_string)
        """
        # Split by whitespace
        parts = input_str.split(None, 1)
        
        if len(parts) == 2:
            try:
                num = float(parts[0])
                return num, parts[1]
            except ValueError:
                pass
        
        # Fallback: try to parse as just a number
        try:
            return float(input_str), ''
        except ValueError:
            return 0.0, ''
    
    def _get_unit_data(self, units_str: str) -> Dict[str, Any]:
        """Get unit conversion data."""
        # Try to get from context (same as FormulaWithUnits)
        if hasattr(self.context, 'isUnit') and self.context.isUnit(units_str):
            if hasattr(self.context, 'constants'):
                const = self.context.constants.get(units_str)
                if const is not None:
                    return {'factor': float(const), 'name': units_str}
        
        return {'factor': 1.0, 'name': units_str}
    
    def __float__(self) -> float:
        """Convert to float (just the numeric value)."""
        return self.value
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.value} {self.units_str}"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"NumberWithUnits({self.value}, '{self.units_str}')"


def FormulaWithUnits_factory(*args, **kwargs):
    """Factory function for FormulaWithUnits (matches Perl API).
    
    Usage:
        FormulaWithUnits("3*x ft")
        FormulaWithUnits(Formula("3*x"), "ft")
    """
    return FormulaWithUnits(*args, **kwargs)


def NumberWithUnits_factory(*args, **kwargs):
    """Factory function for NumberWithUnits (matches Perl API).
    
    Usage:
        NumberWithUnits("3.5 ft")
        NumberWithUnits(3.5, "ft")
    """
    return NumberWithUnits(*args, **kwargs)


__all__ = [
    'FormulaWithUnits',
    'NumberWithUnits',
    'FormulaWithUnits_factory',
    'NumberWithUnits_factory',
]

