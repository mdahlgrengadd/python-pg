"""
LinearRelation Parser

Provides LinearRelation for checking linear relationships between variables.

Based on WeBWorK's PG macro libraries (parserLinearRelation.pl).

This macro library provides a context LinearRelation with a LinearRelation
Math Object using =, <, >, <=, >=, or !=.

Activate the context with:
    Context("LinearRelation")

Use LinearRelation(formula), Formula(formula), or Compute(formula) to
create a LinearRelation object using a string formula. Alternatively, use
LinearRelation(vector, point, sign) where vector is the normal vector and
point is a point on the plane. Or use LinearRelation(vector, real, sign)
where real is the dot product of any point in the plane with the normal vector.
"""

from typing import Any, Callable, Dict, Optional, Tuple
from copy import deepcopy


class LinearRelation:
    """
    LinearRelation MathObject for checking linear relationships.
    
    Supports equations and inequalities: =, <, >, <=, >=, !=
    
    Reference: macros/parsers/parserLinearRelation.pl
    """
    
    _initialized = False
    
    @classmethod
    def Init(cls):
        """
        Initialize the LinearRelation context.
        
        Creates a context based on Vector context with operators for
        linear relations (equations and inequalities).
        
        Reference: macros/parsers/parserLinearRelation.pl lines 89-168
        """
        if cls._initialized:
            return
        
        from ...math.context import Context, get_context, _contexts
        
        # Get Vector context as base
        vector_context = get_context('Vector')
        context = vector_context.copy(name='LinearRelation')
        
        # Set context properties
        context.name = 'LinearRelation'
        if 'precedence' not in context._storage:
            context._storage['precedence'] = {}
        if 'special' not in context._storage['precedence']:
            # Get special precedence from Vector context or default
            context._storage['precedence']['special'] = 0
        context._storage['precedence']['LinearRelation'] = context._storage['precedence']['special']
        
        if 'value' not in context._storage:
            context._storage['value'] = {}
        context._storage['value']['LinearRelation'] = 'LinearRelation'
        context._storage['value']['Formula'] = 'LinearRelation::formula'
        
        # Remove < from parentheses (line 95)
        context.parens.remove('<')
        
        # Set standardForm flag (default 0)
        context.flags.set(standardForm=0)
        
        # Register operators
        operators = [
            ('=', {
                'kind': 'eq',
                'eval': lambda a, b: a == b,
                'string': ' = ',
            }),
            ('<', {
                'kind': 'lt',
                'eval': lambda a, b: a < b,
                'string': ' < ',
            }),
            ('>', {
                'kind': 'gt',
                'eval': lambda a, b: a > b,
                'reverse': 'lt',
                'string': ' > ',
            }),
            ('<=', {
                'kind': 'le',
                'eval': lambda a, b: a <= b,
                'TeX': r'\le',
                'string': ' <= ',
                'alternatives': ['=<', '\u2264'],  # ≤
            }),
            ('>=', {
                'kind': 'ge',
                'eval': lambda a, b: a >= b,
                'reverse': 'le',
                'TeX': r'\ge',
                'string': ' >= ',
                'alternatives': ['=>', '\u2265'],  # ≥
            }),
            ('!=', {
                'kind': 'ne',
                'eval': lambda a, b: a != b,
                'TeX': r'\ne',
                'string': ' != ',
                'alternatives': ['<>', '\u2260'],  # ≠
            }),
        ]
        
        for op, config in operators:
            context.operators.add(
                op,
                precedence=0.5,
                associativity='left',
                type='bin',
                class_name='LinearRelation::inequality',
                formulaClass='LinearRelation',
                **config
            )
        
        # Register context globally
        _contexts['LinearRelation'] = context
        
        cls._initialized = True
    
    def __init__(self, *args: Any, **options: Any):
        """
        Initialize LinearRelation with arguments.
        
        Supports multiple constructor forms:
        - LinearRelation(formula_string) - Parse string formula
        - LinearRelation(vector, point, sign) - From normal vector and point
        - LinearRelation(vector, real, sign) - From normal vector and dot product
        
        Args:
            *args: Constructor arguments
            **options: Additional options
        """
        # Ensure context is initialized
        self.__class__.Init()
        
        from ...math.context import get_context
        from ...math.formula import Formula
        from ...math.geometric import Vector, Point
        from ...math.numeric import Real
        
        context = get_context('LinearRelation')
        self.context = context
        
        # Handle different constructor forms
        if len(args) == 0:
            raise ValueError("LinearRelation requires at least one argument")
        
        # If single argument and it's already a LinearRelation, return it
        if len(args) == 1 and isinstance(args[0], LinearRelation):
            # Copy attributes
            self.N = args[0].N
            self.d = args[0].d
            self.plane = args[0].plane
            self.original_formula = getattr(args[0], 'original_formula', None)
            self.original_formula_latex = getattr(args[0], 'original_formula_latex', None)
            self.options = {**getattr(args[0], 'options', {}), **options}
            return
        
        N = args[0] if len(args) > 0 else None
        p = args[1] if len(args) > 1 else None
        bop = args[2] if len(args) > 2 else None
        
        self.options = options
        
        if p is not None:
            # Constructor form: LinearRelation(vector, point/real, sign)
            # Make sure N is a Vector
            if not isinstance(N, Vector):
                if isinstance(N, (list, tuple)):
                    N = Vector(N, context=context)
                else:
                    N = Vector([N], context=context)
            
            # Make sure p is a Point or Real
            if isinstance(p, (list, tuple)):
                p = Point(p, context=context)
            elif not isinstance(p, Point):
                # Check if it's a Vector (convert to Point)
                if isinstance(p, Vector):
                    p = Point(p.components, context=context)
                else:
                    # Treat as Real (dot product value)
                    p = Real(p) if not isinstance(p, Real) else p
            
            # Calculate d (constant on right side)
            if isinstance(p, Real):
                d = p
            else:
                # d = p . N (dot product of point with normal vector)
                # Convert Point to Vector for dot product
                if isinstance(p, Point):
                    p_vec = Vector(p.coords, context=context)
                else:
                    p_vec = p
                
                # Calculate dot product
                dot_sum = 0
                for i in range(min(len(p_vec.components), len(N.components))):
                    p_val = p_vec.components[i].to_python() if hasattr(p_vec.components[i], 'to_python') else float(p_vec.components[i])
                    n_val = N.components[i].to_python() if hasattr(N.components[i], 'to_python') else float(N.components[i])
                    dot_sum += p_val * n_val
                d = Real(dot_sum)
            
            # Build left side: N[0]*x + N[1]*y + N[2]*z + ...
            variables = sorted(context.variables.list())
            terms = []
            for i, var in enumerate(variables):
                if i < len(N.components):
                    coeff = N.components[i]
                    coeff_str = coeff.to_string() if hasattr(coeff, 'to_string') else str(coeff)
                    terms.append(f"{coeff_str}*{var}")
            
            if not terms:
                raise ValueError("No variables in context for LinearRelation")
            
            leftside_str = " + ".join(terms)
            bop = bop if bop is not None else '='
            d_str = d.to_string() if hasattr(d, 'to_string') else str(d)
            plane_str = f"{leftside_str} {bop} {d_str}"
            
            # Create formula
            plane = Formula(plane_str, variables=variables, context=context)
            plane = plane.reduce() if hasattr(plane, 'reduce') else plane
            
        else:
            # Constructor form: LinearRelation(formula_string)
            # Determine normal vector and d value from the equation
            plane = N
            if not isinstance(plane, Formula):
                plane = Formula(str(plane), context=context)
            
            # Verify it's a relation type (has operator)
            # For now, we'll assume it's a relation if it contains =, <, >, etc.
            plane_str = str(plane)
            if not any(op in plane_str for op in ['=', '<', '>', '<=', '>=', '!=']):
                raise ValueError("Your formula doesn't look like a linear relation")
            
            # Extract variables
            variables = sorted(context.variables.list())
            if not variables:
                # Try to extract from formula
                variables = plane.variables if hasattr(plane, 'variables') else []
                if not variables:
                    # Default to x, y, z if context doesn't have variables
                    variables = ['x', 'y', 'z']
                    for var in variables:
                        if var not in context.variables.list():
                            context.variables.add(var, 'Real')
            
            # Find coefficients by evaluating at test points
            # Use SymPy to extract coefficients if available
            N = None
            d = None
            bop = None
            
            try:
                import sympy as sp
                from sympy.parsing.sympy_parser import parse_expr
                
                # Parse the formula (assume form: left op right)
                # Split by operator (check longer operators first)
                for op in ['<=', '>=', '!=', '=', '<', '>']:
                    if op in plane_str:
                        parts = plane_str.split(op, 1)
                        if len(parts) == 2:
                            left_str = parts[0].strip()
                            right_str = parts[1].strip()
                            bop = op
                            
                            # Parse both sides
                            try:
                                # Replace ^ with ** for SymPy
                                left_expr = parse_expr(left_str.replace('^', '**'), 
                                                      local_dict={v: sp.Symbol(v) for v in variables})
                                right_expr = parse_expr(right_str.replace('^', '**'),
                                                       local_dict={v: sp.Symbol(v) for v in variables})
                                
                                # Get difference: left - right
                                diff_expr = left_expr - right_expr
                                
                                # Extract coefficients for each variable
                                coeffs = []
                                for var in variables:
                                    var_sym = sp.Symbol(var)
                                    coeff = diff_expr.coeff(var_sym)
                                    # Convert to float, handling symbolic expressions
                                    if coeff.is_number:
                                        coeffs.append(float(coeff))
                                    else:
                                        # Try to simplify and evaluate
                                        try:
                                            coeffs.append(float(sp.simplify(coeff)))
                                        except:
                                            coeffs.append(0.0)
                                
                                # Get constant term (evaluate at zero for all variables)
                                constant = diff_expr.subs({sp.Symbol(v): 0 for v in variables})
                                if constant.is_number:
                                    d = Real(-float(constant))
                                else:
                                    try:
                                        d = Real(-float(sp.simplify(constant)))
                                    except:
                                        d = Real(0)
                                
                                N = Vector(coeffs, context=context)
                                break
                            except Exception as e:
                                # Try next operator
                                continue
                
                if N is None or d is None:
                    raise ValueError("Could not extract coefficients from formula")
                    
            except ImportError:
                # SymPy not available - use fallback
                # This is a simplified implementation that may not work for all cases
                raise ValueError("SymPy is required for LinearRelation coefficient extraction")
            
            # Store original formula
            self.original_formula = str(plane)
            self.original_formula_latex = plane.to_tex() if hasattr(plane, 'to_tex') else str(plane)
        
        # Store normal vector and constant
        self.N = N
        self.d = d
        self.plane = plane
        # Extract operator from plane string if not set
        if 'bop' not in locals() or bop is None:
            plane_str = str(plane)
            for op in ['<=', '>=', '!=', '=', '<', '>']:
                if op in plane_str:
                    bop = op
                    break
            else:
                bop = '='
        self.bop = bop
        self.type = 'Relation'  # Mark as Relation type
    
    @property
    def reduce(self) -> 'LinearRelation':
        """
        Reduce/simplify the linear relation.
        
        Returns:
            Self for method chaining
        
        Note: This is a property (not a method) to match Perl MathObjects behavior
        where ->reduce and ->reduce() are equivalent.
        """
        if hasattr(self.plane, 'reduce'):
            # If plane has reduce as property, call it
            if isinstance(self.plane.reduce, property):
                reduced = self.plane.reduce
            else:
                reduced = self.plane.reduce() if callable(self.plane.reduce) else self.plane.reduce
            self.plane = reduced
        return self
    
    def compare(self, other: Any, tolerance: float = 0.001) -> int:
        """
        Compare two LinearRelation objects.
        
        Returns:
            0 if equivalent, 1 if different
        
        Reference: macros/parsers/parserLinearRelation.pl lines 231-276
        """
        from ...math.geometric import Vector
        from ...math.numeric import Real
        
        # Convert other to LinearRelation if needed
        if not isinstance(other, LinearRelation):
            try:
                other = LinearRelation(other, context=self.context)
            except Exception:
                return 1  # Not equivalent
        
        lN, ld = self.N, self.d
        rN, rd = other.N, other.d
        
        # Get inequality types from bop (binary operator)
        # Map operators to kinds
        op_to_kind = {
            '=': 'eq',
            '<': 'lt',
            '>': 'gt',
            '<=': 'le',
            '>=': 'ge',
            '!=': 'ne'
        }
        op_to_reverse = {
            '>': 'lt',
            '>=': 'le'
        }
        
        lbop = getattr(self, 'bop', '=')
        rbop = getattr(other, 'bop', '=')
        ltype = op_to_kind.get(lbop, 'eq')
        rtype = op_to_kind.get(rbop, 'eq')
        lrev = op_to_reverse.get(lbop)
        rrev = op_to_reverse.get(rbop)
        
        # Handle true/false relations (no type) by checking at zero
        zero = Vector([0] * len(lN.components), context=self.context)
        if not ltype:
            ltype = 'eq' if self.check_at(zero) else 'ne'
        if not rtype:
            rtype = 'eq' if other.check_at(zero) else 'ne'
        
        # Reverse inequalities if needed (gt/ge → lt/le)
        if lrev:
            lN = Vector([-c.to_python() for c in lN.components], context=self.context)
            ld = Real(-ld.to_python() if hasattr(ld, 'to_python') else -float(ld))
            ltype = lrev
        if rrev:
            rN = Vector([-c.to_python() for c in rN.components], context=self.context)
            rd = Real(-rd.to_python() if hasattr(rd, 'to_python') else -float(rd))
            rtype = rrev
        
        # Check if inequality types match
        if ltype != rtype:
            return 1
        
        # Handle zero vectors (both sides constant)
        zero_vec = Vector([0] * len(lN.components), context=self.context)
        if lN.compare(zero_vec, tolerance) and rN.compare(zero_vec, tolerance):
            ltruth = self.check_at(zero)
            rtruth = other.check_at(zero)
            # Return 0 if both true or both false, 1 otherwise
            return 0 if (ltruth and rtruth) or (not ltruth and not rtruth) else 1
        
        # Check if normal vectors are parallel
        # For inequalities, require same direction (second parameter to isParallel)
        same_direction = (ltype not in ('eq', 'ne'))
        
        # Check if vectors are parallel
        # Note: is_parallel signature is is_parallel(other, tolerance)
        # For same direction check, we need to verify the vectors point in same direction
        is_parallel = lN.is_parallel(rN, tolerance)
        
        if not is_parallel:
            # Vectors are not parallel - compare directly
            # Return 1 (different) if any component differs
            for i in range(min(len(lN.components), len(rN.components))):
                l_val = lN.components[i].to_python() if hasattr(lN.components[i], 'to_python') else float(lN.components[i])
                r_val = rN.components[i].to_python() if hasattr(rN.components[i], 'to_python') else float(rN.components[i])
                if abs(l_val - r_val) > tolerance:
                    return 1
            # All components match
            return 0
        
        # Vectors are parallel - check if they represent the same plane
        # Compare constants: rd * lN <=> ld * rN
        rd_val = rd.to_python() if hasattr(rd, 'to_python') else float(rd)
        ld_val = ld.to_python() if hasattr(ld, 'to_python') else float(ld)
        
        # Handle zero cases
        if abs(rd_val) < tolerance and abs(ld_val) < tolerance:
            return 0  # Both zero - same plane
        
        if abs(ld_val) < tolerance:
            # ld is zero, check if rd is also zero (within tolerance)
            return 0 if abs(rd_val) < tolerance else 1
        
        if abs(rd_val) < tolerance:
            # rd is zero, check if ld is also zero (within tolerance)
            return 0 if abs(ld_val) < tolerance else 1
        
        # Check if rd/ld ratio matches the vector ratio
        # Since vectors are parallel, check if rd/lN_mag = ld/rN_mag
        # Simplified: check if rd * rN_mag ≈ ld * lN_mag
        lN_mag = sum(c.to_python()**2 for c in lN.components if hasattr(c, 'to_python'))**0.5
        rN_mag = sum(c.to_python()**2 for c in rN.components if hasattr(c, 'to_python'))**0.5
        
        if abs(lN_mag) < tolerance or abs(rN_mag) < tolerance:
            # One vector is zero - already handled above
            return 0
        
        # Compare rd * lN_mag with ld * rN_mag
        left_val = rd_val * lN_mag
        right_val = ld_val * rN_mag
        
        if abs(left_val - right_val) < tolerance:
            return 0
        else:
            return 1
    
    def check_at(self, point: Any) -> bool:
        """
        Check if a point satisfies the relation.
        
        Args:
            point: Point, Vector, or array/list of coordinates
        
        Returns:
            True if point satisfies the relation, False otherwise
        
        Reference: macros/parsers/parserLinearRelation.pl lines 313-328
        """
        from ...math.geometric import Vector, Point
        from ...math.numeric import Real
        
        # Convert to Vector if needed
        if isinstance(point, (list, tuple)):
            point = Vector(point, context=self.context)
        elif isinstance(point, Point):
            # Convert Point to Vector for dot product
            point = Vector(point.coords, context=self.context)
        elif not isinstance(point, Vector):
            raise ValueError("check_at argument must be array, Point, or Vector")
        
        # Get variables from context
        variables = sorted(self.context.variables.list())
        n = len(variables)
        
        # Verify dimension
        point_dim = len(point.components) if hasattr(point, 'components') else len(point)
        if n != point_dim:
            raise ValueError(
                f"The context for this linear relation has {n} variables: "
                f"{', '.join(variables)}, so a point to check at must also have {n} entries"
            )
        
        # Evaluate relation: N . point - d op 0
        # Calculate dot product N . point
        dot_product = 0
        for i in range(min(len(self.N.components), len(point.components))):
            n_val = self.N.components[i].to_python() if hasattr(self.N.components[i], 'to_python') else float(self.N.components[i])
            p_val = point.components[i].to_python() if hasattr(point.components[i], 'to_python') else float(point.components[i])
            dot_product += n_val * p_val
        
        d_val = self.d.to_python() if hasattr(self.d, 'to_python') else float(self.d)
        
        # Get operator from plane or bop
        bop = getattr(self, 'bop', '=')
        
        # Evaluate based on operator
        if bop == '=':
            return abs(dot_product - d_val) < 0.001
        elif bop == '<':
            return dot_product < d_val
        elif bop == '>':
            return dot_product > d_val
        elif bop == '<=':
            return dot_product <= d_val
        elif bop == '>=':
            return dot_product >= d_val
        elif bop == '!=':
            return abs(dot_product - d_val) >= 0.001
        else:
            # Default to equality
            return abs(dot_product - d_val) < 0.001
    
    def string(self) -> str:
        """
        Return string representation.
        
        If standardForm flag is False and original_formula exists, return original.
        Otherwise, return standard form.
        
        Reference: macros/parsers/parserLinearRelation.pl lines 295-302
        """
        if not self.context.flags.get('standardForm', False) and hasattr(self, 'original_formula') and self.original_formula:
            return self.original_formula
        else:
            return str(self.plane) if hasattr(self, 'plane') else str(self)
    
    def to_string(self) -> str:
        """Alias for string() for MathValue compatibility."""
        return self.string()
    
    def TeX(self) -> str:
        """
        Return LaTeX representation.
        
        If standardForm flag is False and original_formula_latex exists, return original.
        Otherwise, return standard form.
        
        Reference: macros/parsers/parserLinearRelation.pl lines 304-311
        """
        if not self.context.flags.get('standardForm', False) and hasattr(self, 'original_formula_latex') and self.original_formula_latex:
            return self.original_formula_latex
        else:
            return self.plane.to_tex() if hasattr(self, 'plane') and hasattr(self.plane, 'to_tex') else str(self)
    
    def to_tex(self) -> str:
        """Alias for TeX() for MathValue compatibility."""
        return self.TeX()
    
    def typeMatch(self, other: Any, ans: Any = None) -> bool:
        """
        Type matching for answer checking.
        
        Only compare two relations.
        
        Reference: macros/parsers/parserLinearRelation.pl lines 289-293
        """
        if not isinstance(self, LinearRelation):
            return isinstance(other, LinearRelation) and getattr(other, 'type', None) == 'Relation'
        return isinstance(other, LinearRelation) and getattr(self, 'type', 'Relation') == getattr(other, 'type', 'Relation')
    
    def isConstant(self) -> bool:
        """
        Check if constant.
        
        Returns False if type is Relation, otherwise calls parent.
        
        Reference: macros/parsers/parserLinearRelation.pl lines 330-334
        """
        if getattr(self, 'type', None) == 'Relation':
            return False
        # For now, return False (would call parent in full implementation)
        return False
    
    def cmp(self, **options: Any) -> Callable:
        """
        Return an answer checker function.
        
        Args:
            **options: Checker options
        
        Returns:
            Function that checks student answer
        """
        from ...math.answer_checker import AnswerChecker
        
        class LinearRelationChecker(AnswerChecker):
            def __init__(self, correct: LinearRelation, **opts):
                super().__init__(correct, **opts)
                self.correct = correct
            
            def check(self, student_answer: str) -> Dict[str, Any]:
                try:
                    # Parse student answer as LinearRelation
                    student = LinearRelation(student_answer, context=self.correct.context)
                    
                    # Compare using compare method
                    result = self.correct.compare(student, self.options.get('tolerance', 0.001))
                    is_correct = (result == 0)
                    
                    return {
                        'correct': is_correct,
                        'score': 1.0 if is_correct else 0.0,
                        'message': '' if is_correct else 'Incorrect linear relation'
                    }
                except Exception as e:
                    return {
                        'correct': False,
                        'score': 0.0,
                        'message': f'Error checking answer: {str(e)}'
                    }
        
        return LinearRelationChecker(self, **options)
    
    def cmp_class(self) -> str:
        """Return class name for error messages."""
        return 'a Linear Relation'
    
    def showClass(self) -> str:
        """Alias for cmp_class()."""
        return self.cmp_class()
    
    def __str__(self) -> str:
        """String representation."""
        return self.string()
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"LinearRelation({self.string()})"


# Initialize on import
LinearRelation.Init()


__all__ = ['LinearRelation']
