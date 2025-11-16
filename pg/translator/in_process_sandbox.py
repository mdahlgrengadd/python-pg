"""
In-Process Sandbox for PG Code Execution.

Provides safe in-process execution with:
- Restricted builtins (no eval, exec, __import__, open, etc.)
- Macro function integration
- Direct access to evaluator objects (no serialization)
- Timeout protection

This matches the Perl PG Safe compartment architecture.
"""

from __future__ import annotations

import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Note: Packages have been refactored into subdirectories under pg/
# Do NOT add OLD_DEPRECATED_packages to sys.path as it would conflict with the new pg/ structure
# All macros are now properly exported from the pg.* namespace

from pg.parser import Context

# Import macro modules
try:
    # Phase 0: Already migrated modules
    from pg.macros.graph.parser_graphtool import GraphTool
    from pg.macros.math.draggable_proof import DraggableProof
    from pg.macros.ui.nice_tables import LayoutTable

    # Phase 1: Critical Graphing Infrastructure
    from pg.macros.graph.pg_graph import init_graph, add_functions, Plot, WWPlot, Label, Fun
    from pg.macros.graph.tikz_image import createTikZImage, TikZImage
    from pg.macros.graph.latex_image import createLaTeXImage, LaTeXImage

    # Phase 2: Essential Parsers
    from pg.macros.parsers.parser_formula_with_units import NumberWithUnits
    from pg.macros.parsers.parser_implicit_plane import ImplicitPlane
    from pg.macros.parsers.parser_parametric_line import ParametricLine
    from pg.macros.parsers.parser_implicit_equation import ImplicitEquation
    from pg.macros.parsers.parser_solution_for import SolutionFor

    # Phase 3: Problem Structure & UI
    from pg.macros.ui.scaffold import Scaffold, Section

    # Phase 4: Math Utilities
    from pg.macros.math.vector_utils import norm, unit, Line
    from pg.macros.math.auxiliary_functions import (
        Round, nicestring, randomPerson, non_zero_point3D, non_zero_vector3D
    )

    # Phase 5: Answer Systems
    from pg.macros.answers.answer_composition import COMPOSITION_ANS
    from pg.macros.answers.unordered_answer import UNORDERED_ANS
    from pg.macros.answers.answer_hints import AnswerHints
    from pg.macros.parsers.parser_checkbox_list import CheckboxList

    # Phase 6: Advanced 3D Graphics
    from pg.macros.graph.vector_field_3d import VectorField3D
    from pg.macros.graph.live_graphics_3d import Graph3D_function as Graph3D

    # Phase 7: Interactive Features
    from pg.macros.math.draggable_proof import DraggableProof
    from pg.macros.math.draggable_subsets import DraggableSubsets
    from pg.macros.graph.parser_graphtool import GraphTool

    # Phase 8: Context & Grading
    from pg.macros.contexts.limited_powers import LimitedPowers
    from pg.macros.parsers.parser_assignment import AssignmentParser, Assignment as parser_Assignment
    from pg.macros.parsers.parser_function import parserFunction
    from pg.macros.core.pg_graders import install_problem_grader, custom_problem_grader_fluid

    # Phase 9: Core Utilities
    from pg.macros.core.pgml_utils import tag, helpLink
    from pg.macros.math.statistics_utils import (
        linear_regression, sample_correlation
    )

    # Phase 10: Fallback Cleanup & Array Utilities
    from pg.macros.core.fallback_utilities import (
        new_match_list, pop_up_list_print_q, undef
    )
    from pg.macros.core.array_utilities import (
        splice, push, pop, shift, unshift, scalar
    )
    from pg.macros.core.pg_standard import (
        random_subset, shuffle
    )

    # Phase 11: Final Stubs - Context & Parser Macros
    from pg.macros.answers.multi_answer import MultiAnswer
    from pg.macros.parsers.parser_popup import PopUp, DropDown, DropDownTF, RadioButtons
    from pg.macros.parsers.parser_radio_multianswer import RadioMultiAnswer
    from pg.macros.parsers.parser_linear_relation import LinearRelation
    from pg.macros.parsers.parser_difference_quotient import DifferenceQuotient
    from pg.macros.parsers.parser_special_trig import specialRadical, specialAngle
    from pg.macros.math.statistics_utils import stats_mean, stats_sd, stats_SX_SXX

    _MACROS_AVAILABLE = True
except ImportError:
    _MACROS_AVAILABLE = False


@dataclass
class ExecutionResult:
    """Result from in-process execution."""

    success: bool
    """Whether execution succeeded"""

    output_text: str
    """Accumulated problem text"""

    answers: dict[str, Any]
    """Answer evaluators by name"""

    solution_text: str | None = None
    """Solution text if any"""

    hint_text: str | None = None
    """Hint text if any"""

    errors: str = ""
    """Error messages"""

    variables: dict[str, Any] = field(default_factory=dict)
    """Problem variables"""


class TimeoutError(Exception):
    """Execution timeout exceeded."""
    pass


class PerlList(list):
    """
    A Python list that behaves like a Perl array.

    Perl arrays auto-vivify when you assign to an index beyond the current size.
    This class replicates that behavior for compatibility with converted Perl code.

    Example:
        >>> arr = PerlList()
        >>> arr[5] = 'value'  # No IndexError - extends list
        >>> arr
        PerlList([None, None, None, None, None, 'value'])
    """

    def __setitem__(self, index, value):
        """
        Set an item, auto-extending the list if necessary.

        If index is greater than the current length, fill intermediate positions
        with None (like Perl's undef).
        """
        if isinstance(index, slice):
            # Handle slice assignment normally
            super().__setitem__(index, value)
        else:
            # Handle integer index with auto-vivification
            if index < 0:
                # Negative indices work normally
                super().__setitem__(index, value)
            else:
                # Extend list with None values if needed
                while len(self) <= index:
                    self.append(None)
                super().__setitem__(index, value)


class InProcessSandbox:
    """
    Safe in-process code execution sandbox.

    Executes PG code in a restricted namespace with:
    - Limited builtins (no dangerous functions)
    - Macro functions available
    - PGEnvironment integration
    - Timeout protection (Unix only)
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize sandbox.

        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout
        self.namespace: dict[str, Any] = {}
        self._pg_environment = None
        self._setup_safe_namespace()

    def _setup_safe_namespace(self) -> None:
        """Setup namespace with safe builtins only."""
        # Safe built-in functions
        safe_builtins = {
            # Type constructors
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': PerlList,  # Use PerlList instead of list for Perl-like array behavior
            'tuple': tuple,
            'dict': dict,
            'set': set,
            # Math functions
            'abs': abs,
            'round': round,
            'pow': pow,
            'min': min,
            'max': max,
            'sum': sum,
            # Utilities
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'reversed': reversed,
            'all': all,
            'any': any,
            # String functions
            'chr': chr,
            'ord': ord,
            # Type checking
            'isinstance': isinstance,
            'issubclass': issubclass,
            'type': type,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
            # Constants
            'True': True,
            'False': False,
            'None': None,
            # Exceptions (needed for try/except)
            'Exception': Exception,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'KeyError': KeyError,
            'IndexError': IndexError,
            'AttributeError': AttributeError,
            'ZeroDivisionError': ZeroDivisionError,
        }

        # Allow controlled __import__ for specific modules
        import builtins
        original_import = builtins.__import__

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Allow importing only safe modules."""
            # Allow pg.* namespace packages (refactored structure)
            if name.startswith('pg.') or name == 'pg':
                return original_import(name, globals, locals, fromlist, level)
            # Allow math, random, and re (already in namespace but allow re-import)
            if name in ('math', 'random', 're'):
                return original_import(name, globals, locals, fromlist, level)
            # Block everything else
            raise ImportError(f"Import of '{name}' is not allowed in sandbox")

        safe_builtins['__import__'] = safe_import

        # Restricted builtins dict
        self.namespace['__builtins__'] = safe_builtins

        # Add PerlList for Perl-like array behavior
        self.namespace['PerlList'] = PerlList

        # Import safe modules
        import math
        import random
        import re
        self.namespace['math'] = math
        self.namespace['random'] = random
        self.namespace['re'] = re

        # Add common mathematical constants
        self.namespace['pi'] = math.pi
        self.namespace['e'] = math.e

        # Add common mathematical functions (for bare function calls)
        self.namespace['sqrt'] = math.sqrt
        self.namespace['sin'] = math.sin
        self.namespace['cos'] = math.cos
        self.namespace['tan'] = math.tan
        self.namespace['asin'] = math.asin
        self.namespace['acos'] = math.acos
        self.namespace['atan'] = math.atan
        self.namespace['arcsin'] = math.asin  # Alias
        self.namespace['arccos'] = math.acos  # Alias
        self.namespace['arctan'] = math.atan  # Alias
        self.namespace['exp'] = math.exp
        self.namespace['log'] = math.log
        self.namespace['ln'] = math.log  # Alias
        self.namespace['abs'] = abs

        # Load MathObjects
        self._load_mathobjects()

        # Load all macros (static loading)
        # Preprocessor converts loadMacros() to Python imports
        self._load_pg_core()
        self._load_pg_basic_macros()
        self._load_pg_answer_macros()
        self._load_context_macros()
        self._load_parser_macros()
        self._load_statistics_macros()

    def _load_mathobjects(self) -> None:
        """Load MathObjects framework into namespace."""
        try:
            # Import MathObjects
            # Use pg.math for all MathObjects (has full interval/inequality support)
            from pg.math.context import get_context as _get_context
            from pg.math.compute import Compute as _Compute
            from pg.math import Formula, Real
            from pg.math import FormulaUpToConstant
            from pg.math import Complex as _Complex, List as _List, Point, Vector, Interval, Set, Fraction, String as _String

            # Context function that delegates to pg_math
            def Context(name=None):
                """
                Context function - delegates to pg_math.context.get_context.

                When called without arguments, returns the current context.
                When called with a name, switches to that context and returns it.

                Examples:
                    ctx = Context()           # Get current context
                    ctx = Context('Numeric')  # Switch to Numeric context
                    Context().variables.are(x='Real', y='Real')
                """
                ctx = _get_context(name)

                # Sync context constants to namespace
                # This makes i, j, k (and other context-specific constants) available
                for const_name in ['i', 'j', 'k']:
                    if const_name in ctx.constants:  # Uses __contains__ method
                        self.namespace[const_name] = ctx.constants.get(const_name)

                return ctx

            # Compute function that delegates to pg_math
            def Compute(expr):
                """Compute function - delegates to pg_math.compute.Compute."""
                # Get the current context and pass it to Compute
                # Use the context wrapper to ensure it's consistent
                _current_context = _get_context()  # This uses the namespace's Context function
                return _Compute(expr, context=_current_context)

            # Wrapper for Complex that handles list arguments (Perl compatibility)
            def Complex(real, imag=0, **kwargs):
                """Complex wrapper that handles list/array arguments like Perl."""
                if isinstance(real, (list, tuple)):
                    # Unpack list: Complex([a, b]) → Complex(a, b)
                    if len(real) >= 2:
                        return _Complex(real[0], real[1], **kwargs)
                    elif len(real) == 1:
                        return _Complex(real[0], 0, **kwargs)
                    else:
                        return _Complex(0, 0, **kwargs)
                elif isinstance(real, str):
                    # String form: Complex("2-4i") - parse it
                    # For now, just pass to _Complex and let it handle or fail gracefully
                    import re
                    match = re.match(
                        r'([+-]?\d+(?:\.\d+)?)\s*([+-])\s*(\d+(?:\.\d+)?)i', real.replace(' ', ''))
                    if match:
                        r = float(match.group(1))
                        sign = match.group(2)
                        i = float(match.group(3))
                        if sign == '-':
                            i = -i
                        return _Complex(r, i, **kwargs)
                return _Complex(real, imag, **kwargs)

            # Wrapper for List that handles variadic arguments (Perl compatibility)
            def List(*args, **kwargs):
                """List wrapper that handles variadic arguments like Perl."""
                if len(args) == 1 and isinstance(args[0], (list, tuple)):
                    # Single list argument: List([1, 2, 3])
                    return _List(list(args[0]), **kwargs)
                else:
                    # Multiple arguments: List(1, 2, 3)
                    return _List(list(args), **kwargs)

            # Wrapper for Formula that uses Compute in Units context  
            def Formula_wrapper(expr, **kwargs):
                """Formula wrapper that uses Compute for unit expressions in Units context."""
                # Get current context
                _current_ctx = _get_context()
                
                # ONLY modify behavior for Units context to handle unit expressions
                # For all other contexts, use Formula directly without changes
                if _current_ctx and _current_ctx.name in ('Units', 'LimitedUnits') and isinstance(expr, str):
                    # In Units context, use Compute to handle unit expressions
                    result = _Compute(expr, context=_current_ctx)
                    # Return whatever Compute returns (Formula, FormulaWithUnits, or NumberWithUnits)
                    return result
                
                # For non-Units contexts or non-string expressions, use Formula unchanged
                return Formula(expr, **kwargs)

            # Make available in namespace
            self.namespace['Context'] = Context
            self.namespace['Formula'] = Formula_wrapper
            self.namespace['Real'] = Real
            self.namespace['Complex'] = Complex
            self.namespace['Compute'] = Compute
            self.namespace['FormulaUpToConstant'] = FormulaUpToConstant
            self.namespace['List'] = List
            self.namespace['String'] = _String
            self.namespace['Point'] = Point
            self.namespace['Vector'] = Vector
            self.namespace['Interval'] = Interval
            self.namespace['Set'] = Set
            self.namespace['Fraction'] = Fraction

            # Create imaginary unit i = Complex(0, 1)
            # Also define j and k as aliases for i (engineering notation and vector unit vectors)
            # i, j, k are now context-dependent constants
            # They will be set by Context() when called
            # In Complex context: i is the imaginary unit
            # In Vector context: i, j, k are unit vectors
            # Initialize with Complex defaults for backward compatibility
            self.namespace['i'] = _Complex(0, 1)
            self.namespace['j'] = _Complex(0, 1)
            self.namespace['k'] = _Complex(0, 1)

            # Array and utility functions are now imported from pg.macros modules
            self.namespace['random_subset'] = random_subset
            self.namespace['shuffle'] = shuffle
            self.namespace['new_match_list'] = new_match_list
            self.namespace['pop_up_list_print_q'] = pop_up_list_print_q
            self.namespace['linear_regression'] = linear_regression
            self.namespace['splice'] = splice
            self.namespace['push'] = push
            self.namespace['pop'] = pop
            self.namespace['shift'] = shift
            self.namespace['unshift'] = unshift

        except ImportError as e:
            # pg_macros modules are required for proper operation
            raise ImportError(
                f"Failed to import pg.macros modules. All macro functionality requires "
                f"the pg_macros package to be properly installed. Error: {e}"
            ) from e

    def load_macros(self, *_macro_names: str) -> None:
        """
        Load PG macro modules into namespace.

        Note: With preprocessor-based import conversion, this method is largely
        a no-op since macros are imported at compile time. However, it's kept
        for backward compatibility with code that manually calls loadMacros().

        Args:
            _macro_names: Macro names (ignored - macros imported by preprocessor)
        """
        # All macros are already loaded in _setup_safe_namespace()
        # Preprocessor converts loadMacros() calls to Python imports
        # This method is kept for compatibility but does nothing
        pass

    def _load_pg_core(self) -> None:
        """Load PG core macros."""
        try:
            from pg.macros.core import pg_core

            # Define PGML function (not in pg_core)
            def PGML(pgml_text):
                """Render PGML markup and register answer blanks."""
                from pg.renderer import PGMLRenderer

                # Get current environment
                env = pg_core.get_environment()

                # Create renderer with access to namespace variables
                renderer = PGMLRenderer(variables=self.namespace)

                # Render PGML to HTML and extract answer blanks
                rendered_html, answer_blanks = renderer.render(pgml_text)

                # Register answer blanks in environment
                for ans_name, ans_spec in answer_blanks.items():
                    # For PGML answers, we want to preserve the full spec dict
                    # Store directly in answers_hash to avoid conversion by register_answer
                    if isinstance(ans_spec, dict) and 'evaluator' in ans_spec:
                        # PGML spec format - store full dict directly
                        env.answers_hash[ans_name] = ans_spec
                    else:
                        # Legacy format or evaluator object - use register_answer
                        env.register_answer(ans_name, ans_spec)

                # Return rendered HTML (which TEXT() will append)
                return rendered_html

            def Matrix(*args, **kwargs):
                """Matrix wrapper for compatibility."""
                # Use a special key to avoid recursion with the wrapper itself
                _Matrix = self.namespace.get('_Matrix_class')
                if not _Matrix:
                    try:
                        from pg.math.geometric import Matrix as _Matrix
                        self.namespace['_Matrix_class'] = _Matrix
                    except ImportError:
                        if len(args) == 1 and isinstance(args[0], (list, tuple)):
                            return list(args[0])
                        return list(args)
                if len(args) == 1 and isinstance(args[0], (list, tuple)):
                    return _Matrix(args[0], **kwargs)
                return _Matrix(list(args), **kwargs)

            # Phase 7 functions imported from pg.macros modules

            # Phase 9 functions imported from pg.macros modules

            # Phase 4 stub functions imported from pg.macros modules

            # Phase 6 stub functions imported from pg.macros modules

            # Line stub removed - use pg_macros.math.vector_utils.Line

            # Phase 8 stub functions imported from pg.macros modules

            # Perl compatibility values
            def undef():
                """Stub for Perl's undef - returns None."""
                return None

            def new_match_list(*args, **kwargs):
                """Stub new_match_list - returns stub match list object."""
                class _MatchListStub:
                    def __getattr__(self, name):
                        """Allow any method call - just return self for chaining."""
                        def method(*a, **kw):
                            return self
                        return method

                    def __iter__(self):
                        return iter([])
                return _MatchListStub()

            def pop_up_list_print_q(*args, **kwargs):
                """Stub pop_up_list_print_q - dummy function for printing questions."""
                return ""

            # Helper functions for operators that can work with multiple types
            def pg_concat(left, right):
                """
                Perl dot (.) operator: concatenates two values.
                Works with strings, vectors (dot product), and converts to strings as needed.
                """
                # Check if left is a Vector (has a __mul__ method for dot product)
                if hasattr(left, '__mul__') and hasattr(left, 'value'):
                    # This is likely a Vector - use dot product
                    return left * right
                # Otherwise, treat as string concatenation
                return str(left) + str(right)

            def pg_repeat(left, right):
                """
                Perl repetition (x) operator: repeats a string or computes cross product.
                Works with strings, integers, and vectors (cross product).
                """
                # Check if left is a Vector (has a cross method or cross product support)
                if hasattr(left, '__mul__') and hasattr(left, 'value'):
                    # This is likely a Vector - might have cross product
                    # Try to use cross product if right is also a Vector
                    if hasattr(right, '__mul__') and hasattr(right, 'value'):
                        # Both are Vectors, compute cross product
                        if hasattr(left, 'cross'):
                            return left.cross(right)
                        # Fallback: use * operator if defined
                        return left * right
                # Otherwise, treat as string repetition
                if isinstance(right, int):
                    return str(left) * right
                raise TypeError(f"can't use '{type(right).__name__}' as repeat count")

            # Add pg_core functions to namespace
            # Register core functions
            self.namespace.update({
                'DOCUMENT': pg_core.DOCUMENT,
                'ENDDOCUMENT': pg_core.ENDDOCUMENT,
                'TEXT': pg_core.TEXT,
                'BEGIN_TEXT': pg_core.BEGIN_TEXT,
                'END_TEXT': pg_core.END_TEXT,
                'ANS': pg_core.ANS,
                'NAMED_ANS': pg_core.NAMED_ANS,
                'NEW_ANS_NAME': pg_core.NEW_ANS_NAME,
                'SOLUTION': pg_core.SOLUTION,
                'HINT': pg_core.HINT,
                'COMMENT': pg_core.COMMENT,
                'PGML': PGML,
                'HEADER_TEXT': pg_core.HEADER_TEXT,
                'random': pg_core.random,
                'non_zero_random': pg_core.non_zero_random,
                'list_random': pg_core.list_random,
                'random_coprime': pg_core.random_coprime,
                'loadMacros': pg_core.loadMacros,
                'PGEnvironment': pg_core.PGEnvironment,
                'set_environment': pg_core.set_environment,
                'get_environment': pg_core.get_environment,
                # Helper functions for operators
                'pg_concat': pg_concat,
                'pg_repeat': pg_repeat,
                # Common constants
                'SPACE': ' ',
                # Phase 4: Use imported math utilities directly
                'non_zero_point3D': non_zero_point3D,
                'non_zero_vector3D': non_zero_vector3D,
                'norm': norm,
                'unit': unit,
                'Line': Line,
                'Round': Round,
                'nicestring': nicestring,
                'randomPerson': randomPerson,
                # Other stubs
                'Matrix': Matrix,
                'Graph3D': Graph3D,
                'Plot': Plot,
                'VectorField3D': VectorField3D,
                # Use imported graph macros directly
                'init_graph': init_graph,
                'add_functions': add_functions,
                'createLaTeXImage': createLaTeXImage,
                'createTikZImage': createTikZImage,
                'COMPOSITION_ANS': COMPOSITION_ANS,
                'UNORDERED_ANS': UNORDERED_ANS,
                'AnswerHints': AnswerHints,
                # Use imported modules if available, otherwise use local stubs
                'GraphTool': globals()['GraphTool'] if _MACROS_AVAILABLE else GraphTool,
                # GraphTool object type constants (used in f-strings)
                'point': 'point',
                'solid': 'solid',
                'cubic': 'cubic',
                'line': 'line',
                'circle': 'circle',
                'parabola': 'parabola',
                'vector': 'vector',
                'interval': 'interval',
                'DraggableProof': globals()['DraggableProof'] if _MACROS_AVAILABLE else DraggableProof,
                'DraggableSubsets': DraggableSubsets,
                'CheckboxList': CheckboxList,
                'tag': tag,
                # Phase 2: Use imported parser functions directly
                'NumberWithUnits': NumberWithUnits,
                'ImplicitEquation': ImplicitEquation,
                'SolutionFor': SolutionFor,
                'ParametricLine': ParametricLine,
                'ImplicitPlane': ImplicitPlane,
                # Phase 3: Problem Structure & UI
                'Scaffold': Scaffold,
                'Section': Section,
                'undef': undef,
                # Problem grading
                'install_problem_grader': install_problem_grader,
                'custom_problem_grader_fluid': custom_problem_grader_fluid,
                # Environment dictionary
                'ENV': {},
                # Matching/list utilities
                'new_match_list': new_match_list,
                'pop_up_list_print_q': pop_up_list_print_q,
                # Array manipulation
                'splice': splice,
                'push': push,
                'pop': pop,
                'shift': shift,
                'unshift': unshift,
                'scalar': scalar,
            })

            # Store reference for initialization
            self._pg_core = pg_core
            
            # Add ALPHABET constant (used in matching problems)
            self.namespace['ALPHABET'] = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

        except ImportError:
            # Fallback: provide stub implementations
            self._load_pg_core_stubs()

    def _load_pg_core_stubs(self) -> None:
        """Load stub implementations if pg_core not available."""
        # Simple stub environment
        class StubEnvironment:
            def __init__(self, envir):
                self.data = envir  # Store namespace for PGML variable access
                self.output_array = []
                self.answers_hash = {}
                self.solution_array = []
                self.hint_array = []
                self.pgml_array = []  # For PGML content
                self._answer_counter = 0

            def append_text(self, text):
                self.output_array.append(str(text))

            def new_ans_name(self):
                self._answer_counter += 1
                return f"AnSwEr{self._answer_counter:04d}"

        _env = StubEnvironment(self.namespace)

        def DOCUMENT(): pass
        def ENDDOCUMENT(): pass
        def TEXT(*args): _env.append_text(''.join(str(a) for a in args))
        def BEGIN_TEXT(): return ''
        def END_TEXT(): return ''

        def ANS(*evaluators):
            for ev in evaluators:
                name = _env.new_ans_name()
                _env.answers_hash[name] = ev

        def NAMED_ANS(name, evaluator):
            _env.answers_hash[name] = evaluator

        def NEW_ANS_NAME(): return _env.new_ans_name()
        def SOLUTION(*args): _env.solution_array.extend(str(a) for a in args)
        def HINT(*args): _env.hint_array.extend(str(a) for a in args)
        def COMMENT(*args): pass

        # PGML rendering function
        # NOTE: PGML function is already defined in _load_pg_core() above
        # This stub is deprecated and should not be used
        # The correct PGML function registers answer blanks with the environment

        # Random functions (don't shadow random module)
        import random as _random_module

        def pg_random(a=0, b=1, step=None):
            if step:
                return a + _random_module.randint(0, int((b-a)/step)) * step
            return a + _random_module.random() * (b - a)

        def non_zero_random(a, b, step=None):
            while True:
                v = pg_random(a, b, step)
                if v != 0:
                    return v

        def list_random(*items):
            return _random_module.choice(items)

        def random_coprime(*arrays):
            """Stub for random_coprime."""
            import math
            import random
            from itertools import product
            if not arrays:
                return ()
            # Convert all arrays to lists, expanding list-wrapped ranges

            def expand_array(arr):
                expanded = list(arr)
                if len(expanded) == 1 and isinstance(expanded[0], range):
                    return list(expanded[0])
                return expanded
            list_arrays = [expand_array(arr) for arr in arrays]
            if len(list_arrays) == 1:
                return (list_random(*list_arrays[0]),)
            all_tuples = list(product(*list_arrays))
            coprime = [t for t in all_tuples if math.gcd(
                *[abs(x) for x in t]) == 1]
            return random.choice(coprime) if coprime else tuple([0] * len(list_arrays))

        def loadMacros(*args): pass
        def get_environment(): return _env
        def set_environment(env): pass

        # Geometric stubs

        def non_zero_point3D(*args):
            """Stub for non_zero_point3D - generates non-zero 3D point."""
            # Returns a random 3D point with no zero coordinates
            # Access Point from namespace
            Point = self.namespace.get('Point')
            if Point:
                return Point([non_zero_random(-5, 5), non_zero_random(-5, 5), non_zero_random(-5, 5)])
            # Fallback: return list
            return [non_zero_random(-5, 5), non_zero_random(-5, 5), non_zero_random(-5, 5)]

        def norm(vector):
            """Compute the norm (magnitude/length) of a vector."""
            if hasattr(vector, 'norm'):
                return vector.norm()
            # Fallback for list/tuple
            import math
            return math.sqrt(sum(x**2 for x in vector))

        def unit(vector):
            """Compute the unit vector in the direction of the given vector."""
            if hasattr(vector, 'unit'):
                return vector.unit()
            # Fallback for list/tuple
            import math
            magnitude = math.sqrt(sum(x**2 for x in vector))
            if magnitude == 0:
                raise ValueError("Cannot compute unit vector of zero vector")
            return [x / magnitude for x in vector]

        # Matrix constructor (wrap pg_math Matrix if available)
        def Matrix(*args, **kwargs):
            """Matrix wrapper for compatibility."""
            # Use a special key to avoid recursion with the wrapper itself
            _Matrix = self.namespace.get('_Matrix_class')
            if not _Matrix:
                # Import if not in namespace yet
                try:
                    from pg.math.geometric import Matrix as _Matrix
                    self.namespace['_Matrix_class'] = _Matrix
                except ImportError:
                    # Fallback: return list of lists
                    if len(args) == 1 and isinstance(args[0], (list, tuple)):
                        return list(args[0])
                    return list(args)
            # Handle various calling conventions
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                return _Matrix(args[0], **kwargs)
            return _Matrix(list(args), **kwargs)

        # Graphics stubs

        # Special answer evaluation functions

        # Phase 7 utilities removed - use pg_macros modules

        # Phase 9 utilities removed - use pg_macros modules

        # Phase 4 utilities removed - use pg_macros.math modules

        # Phase 6 utilities removed - use pg_macros.graph modules

        # Perl compatibility values
        def undef():
            """Stub for Perl's undef - returns None."""
            return None

        # Helper functions for operators that can work with multiple types
        def pg_concat(left, right):
            """
            Perl dot (.) operator: concatenates two values.
            Works with strings, vectors (dot product), and converts to strings as needed.
            """
            # Check if left is a Vector (has a __mul__ method for dot product)
            if hasattr(left, '__mul__') and hasattr(left, 'value'):
                # This is likely a Vector - use dot product
                return left * right
            # Otherwise, treat as string concatenation
            return str(left) + str(right)

        def pg_repeat(left, right):
            """
            Perl repetition (x) operator: repeats a string or computes cross product.
            Works with strings, integers, and vectors (cross product).
            """
            # Check if left is a Vector (has a cross method or cross product support)
            if hasattr(left, '__mul__') and hasattr(left, 'value'):
                # This is likely a Vector - might have cross product
                # Try to use cross product if right is also a Vector
                if hasattr(right, '__mul__') and hasattr(right, 'value'):
                    # Both are Vectors, compute cross product
                    if hasattr(left, 'cross'):
                        return left.cross(right)
                    # Fallback: use * operator if defined
                    return left * right
            # Otherwise, treat as string repetition
            if isinstance(right, int):
                return str(left) * right
            raise TypeError(f"can't use '{type(right).__name__}' as repeat count")

        self.namespace.update({
            'DOCUMENT': DOCUMENT,
            'ENDDOCUMENT': ENDDOCUMENT,
            'TEXT': TEXT,
            'PGML': PGML,
            'BEGIN_TEXT': BEGIN_TEXT,
            'END_TEXT': END_TEXT,
            'ANS': ANS,
            'NAMED_ANS': NAMED_ANS,
            'NEW_ANS_NAME': NEW_ANS_NAME,
            'SOLUTION': SOLUTION,
            'HINT': HINT,
            'COMMENT': COMMENT,
            'random': pg_random,
            'non_zero_random': non_zero_random,
            'list_random': list_random,
            'random_coprime': random_coprime,
            'loadMacros': loadMacros,
            'parserFunction': parserFunction,
            'get_environment': get_environment,
            'set_environment': set_environment,
            # Helper functions for operators
            'pg_concat': pg_concat,
            'pg_repeat': pg_repeat,
            # Geometric stubs
            'non_zero_point3D': non_zero_point3D,
            'non_zero_vector3D': non_zero_vector3D,
            'norm': norm,
            'unit': unit,
            'Matrix': Matrix,
            # Graphics (Note: init_graph, add_functions, createLaTeXImage, createTikZImage, Plot
            # are already handled in _load_pg_core() - no duplicates needed here)
            'Graph3D': Graph3D,
            'VectorField3D': VectorField3D,
            'Line': Line,
            # Special answer evaluation
            'COMPOSITION_ANS': COMPOSITION_ANS,
            'UNORDERED_ANS': UNORDERED_ANS,
            'AnswerHints': AnswerHints,
            # Interactive elements
            'DraggableProof': DraggableProof,
            'DraggableSubsets': DraggableSubsets,
            'CheckboxList': CheckboxList,
            # String/HTML utilities
            'tag': tag,
            # Numeric utilities
            'Round': Round,
            # Units
            'NumberWithUnits': NumberWithUnits,
            # Parser utilities
            'ImplicitEquation': ImplicitEquation,
            'SolutionFor': SolutionFor,
            'ParametricLine': ParametricLine,
            'ImplicitPlane': ImplicitPlane,
            # Random utilities
            'randomPerson': randomPerson,
            # Formatting utilities
            'nicestring': nicestring,
            'GraphTool': GraphTool,
            # GraphTool object type constants (used in f-strings)
            'point': 'point',
            'solid': 'solid',
            'cubic': 'cubic',
            'line': 'line',
            'circle': 'circle',
            'parabola': 'parabola',
            'vector': 'vector',
            'interval': 'interval',
            # Perl compatibility
            'undef': undef,
            # Problem grading
            'install_problem_grader': install_problem_grader,
            'custom_problem_grader_fluid': custom_problem_grader_fluid,
            # Array manipulation (from pg.macros modules)
            'splice': splice,
            'push': push,
            'pop': pop,
            'shift': shift,
            'unshift': unshift,
            'scalar': scalar,
            # Constants
            'ALPHABET': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
            # Add a dummy macro loader to suppress warnings
            # Macros are pre-loaded, so this just prevents the warning
            '_macro_loader': type('DummyLoader', (), {'load_macro': lambda self, x: None})(),
        })

        self._stub_env = _env

    def _load_pg_basic_macros(self) -> None:
        """Load PG basic macros."""
        try:
            from pg.macros.core import pg_basic_macros

            # Register basic macro functions
            self.namespace.update({
                'ans_rule': pg_basic_macros.ans_rule,
                'ans_box': pg_basic_macros.ans_box,
                'ans_radio_buttons': pg_basic_macros.ans_radio_buttons,
                'pop_up_list': pg_basic_macros.pop_up_list,
                'NAMED_ANS_RULE': pg_basic_macros.NAMED_ANS_RULE,
                'NAMED_ANS_BOX': pg_basic_macros.NAMED_ANS_BOX,
                'NAMED_ANS_RADIO_BUTTONS': pg_basic_macros.NAMED_ANS_RADIO_BUTTONS,
                'NAMED_POP_UP_LIST': pg_basic_macros.NAMED_POP_UP_LIST,
                'beginproblem': pg_basic_macros.beginproblem,
                'PAR': pg_basic_macros.PAR,
                'BR': pg_basic_macros.BR,
                'BBOLD': pg_basic_macros.BBOLD,
                'EBOLD': pg_basic_macros.EBOLD,
                'BITALIC': pg_basic_macros.BITALIC,
                'EITALIC': pg_basic_macros.EITALIC,
                'BCENTER': pg_basic_macros.BCENTER,
                'ECENTER': pg_basic_macros.ECENTER,
                'BUL': pg_basic_macros.BUL,
                'EUL': pg_basic_macros.EUL,
                'MODES': pg_basic_macros.MODES,
                'image': pg_basic_macros.image,
                'PI': pg_basic_macros.PI,
                'E': pg_basic_macros.E,
            })

        except ImportError:
            # Fallback: provide stub implementations
            self._load_pg_basic_macros_stubs()

    def _load_pg_basic_macros_stubs(self) -> None:
        """Load stub implementations for basic macros."""
        def ans_rule(width=20):
            env = self.namespace.get(
                'get_environment', lambda: self._stub_env)()
            name = env.new_ans_name()
            return f'<input type="text" name="{name}" size="{width}"/>'

        def ans_box(rows=5, cols=20):
            env = self.namespace.get(
                'get_environment', lambda: self._stub_env)()
            name = env.new_ans_name()
            return f'<textarea name="{name}" rows="{rows}" cols="{cols}"></textarea>'

        def beginproblem(): return ""
        def PAR(): return '<p>'
        def BR(): return '<br/>'
        def BBOLD(): return '<strong>'
        def EBOLD(): return '</strong>'
        def BITALIC(): return '<em>'
        def EITALIC(): return '</em>'
        def BCENTER(): return '<div style="text-align:center;">'
        def ECENTER(): return '</div>'
        def BUL(): return '<u>'
        def EUL(): return '</u>'
        def MODES(**kwargs): return kwargs.get('HTML', '')
        def image(filename, **opts): return f'<img src="{filename}"/>'

        import math
        def PI(): return math.pi
        def E(): return math.e

        # Common constants
        SPACE = ' '

        self.namespace.update({
            'ans_rule': ans_rule,
            'ans_box': ans_box,
            'beginproblem': beginproblem,
            'PAR': PAR,
            'BR': BR,
            'SPACE': SPACE,
            'BBOLD': BBOLD,
            'EBOLD': EBOLD,
            'BITALIC': BITALIC,
            'EITALIC': EITALIC,
            'BCENTER': BCENTER,
            'ECENTER': ECENTER,
            'BUL': BUL,
            'EUL': EUL,
            'MODES': MODES,
            'image': image,
            'PI': PI,
            'E': E,
        })

    def _load_pg_answer_macros(self) -> None:
        """Load PG answer checker macros."""
        try:
            from pg.answer.evaluators import NumericEvaluator, StringEvaluator, FormulaEvaluator

            def num_cmp(correct, **options):
                return NumericEvaluator(correct_answer=correct, **options)

            def str_cmp(correct, **options):
                return StringEvaluator(correct_answer=correct, **options)

            def fun_cmp(correct, **options):
                return FormulaEvaluator(correct_answer=correct, **options)

            self.namespace.update({
                'num_cmp': num_cmp,
                'str_cmp': str_cmp,
                'fun_cmp': fun_cmp,
            })

        except ImportError:
            # Fallback: simple stub evaluators
            class StubEvaluator:
                def __init__(self, correct_answer, **options):
                    self.correct_answer = correct_answer
                    self.options = options

                def evaluate(self, student_answer):
                    from pg.answer import AnswerResult
                    try:
                        is_correct = float(student_answer) == float(
                            self.correct_answer)
                        return AnswerResult(
                            is_correct=is_correct,
                            score=1.0 if is_correct else 0.0,
                            student_answer=student_answer,
                            correct_answer=str(self.correct_answer)
                        )
                    except:
                        return AnswerResult(
                            is_correct=False,
                            score=0.0,
                            student_answer=student_answer,
                            correct_answer=str(self.correct_answer),
                            message="Invalid answer format"
                        )

            def num_cmp(correct, **options):
                return StubEvaluator(correct, **options)

            def str_cmp(correct, **options):
                return StubEvaluator(correct, **options)

            def fun_cmp(correct, **options):
                return StubEvaluator(correct, **options)

            self.namespace.update({
                'num_cmp': num_cmp,
                'str_cmp': str_cmp,
                'fun_cmp': fun_cmp,
            })

    def initialize_environment(self, seed: int, context: Context | None = None) -> None:
        """
        Initialize PG environment for problem execution.

        Clears namespace and reinitializes to prevent variable pollution between problems.

        Args:
            seed: Random seed
            context: Mathematical context
        """
        # Clear namespace and reinitialize to prevent variable pollution
        self.namespace.clear()
        self._setup_safe_namespace()

        # IMPORTANT: Reset Context to prevent variable pollution
        # Context is a global singleton that persists between problems
        try:
            from pg.math.context import get_context
            # Force creation of fresh Numeric context
            get_context('Numeric')
        except ImportError:
            pass  # pg_math not available

        # Set random seed
        import random as _random_module
        _random_module.seed(seed)

        # Also set seed on the random module in namespace if it's the real module
        if 'random' in self.namespace and hasattr(self.namespace['random'], 'seed'):
            self.namespace['random'].seed(seed)

        # Set up envir dict that DOCUMENT() will use
        self.namespace['envir'] = {
            'problemSeed': seed,
            'displayMode': 'HTML',
            'showPartialCorrectAnswers': 1,
            'inputs_ref': {},
        }

        # Don't initialize PGEnvironment here - DOCUMENT() will do it
        # Just clear any previous environment
        self._pg_environment = None

        # Also clear pg_core's global environment if we're using it
        if hasattr(self, '_pg_core') and hasattr(self._pg_core, '_pg_environment'):
            self._pg_core._pg_environment = None

    @contextmanager
    def _timeout_context(self):
        """Context manager for timeout protection (Unix only)."""
        if sys.platform == 'win32':
            # Windows doesn't support signal.alarm
            # Use simple time tracking instead
            start_time = time.time()
            yield
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                raise TimeoutError(
                    f"Execution exceeded {self.timeout} seconds")
        else:
            # Unix: use signal.alarm
            def timeout_handler(signum, frame):
                raise TimeoutError(
                    f"Execution exceeded {self.timeout} seconds")

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    def execute(self, code: str, seed: int, context: Context | None = None) -> ExecutionResult:
        """
        Execute PG code in sandbox.

        Args:
            code: Python code to execute
            seed: Random seed
            context: Mathematical context

        Returns:
            ExecutionResult with all problem data
        """
        # Initialize environment
        self.initialize_environment(seed, context)

        errors = ""

        try:
            with self._timeout_context():
                # Compile code with restricted mode
                compiled = compile(code, '<problem>', 'exec')

                # Execute in namespace
                exec(compiled, self.namespace)

        except TimeoutError as e:
            errors = str(e)
        except Exception as e:
            import traceback
            # Collect results from PG environment
            errors = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        # Try to get environment from pg.core global
        try:
            if hasattr(self, '_pg_core'):
                # Use the SAME pg_core instance that was loaded in namespace
                try:
                    pg_env = self._pg_core.get_environment()
                except RuntimeError:
                    # Environment not initialized - get_environment() raises RuntimeError
                    pg_env = None
            elif hasattr(self, '_stub_env'):
                # Use stub environment if pg_core not available
                pg_env = self._stub_env
            else:
                pg_env = self._pg_environment if hasattr(
                    self, '_pg_environment') else None
        except Exception as ex:
            pg_env = None

        if pg_env:
            # Get output from output_array (TEXT() calls, which include PGML content)
            if hasattr(pg_env, 'output_array') and pg_env.output_array:
                output_text = '\n\n'.join(pg_env.output_array)
            else:
                output_text = ''

            answers = dict(pg_env.answers_hash)

            # Also merge answers registered via PGML() function
            # (which registers in the real PGEnvironment via pg_core.get_environment())
            try:
                if hasattr(self, '_pg_core') and hasattr(self._pg_core, '_pg_environment'):
                    real_pg_env = self._pg_core._pg_environment
                    if real_pg_env and hasattr(real_pg_env, 'answers_hash'):
                        # Merge answers from real environment's answers_hash
                        # The answers registered via PGML() function store full spec dicts
                        answers.update(real_pg_env.answers_hash)
            except Exception:
                pass  # If something fails, just use what we have
            solution_text = ''.join(getattr(pg_env, 'solution_array', [])) if hasattr(
                pg_env, 'solution_array') else None
            hint_text = ''.join(getattr(pg_env, 'hint_array', [])) if hasattr(
                pg_env, 'hint_array') else None
        else:
            output_text = ""
            answers = {}
            solution_text = None
            hint_text = None

        # Collect problem variables
        variables = {}
        for key, value in self.namespace.items():
            if not key.startswith('_') and key not in ('__builtins__',):
                # Include:
                # 1. Simple types (int, float, str, bool, list, tuple, dict)
                # 2. MathValue objects (Complex, Real, Vector, Matrix, Formula, etc.)
                # 3. Answer evaluators (objects with evaluate/cmp/check methods)
                # 4. NumberWithUnits objects (for Units context)
                if isinstance(value, (int, float, str, bool, list, tuple, dict)):
                    variables[key] = value
                elif hasattr(value, 'evaluate') or hasattr(value, 'cmp') or hasattr(value, 'check'):
                    # This is likely an answer evaluator (Formula, Real, AnswerChecker, etc.)
                    variables[key] = value
                else:
                    # Try to import MathValue to check instance
                    try:
                        from pg.math.value import MathValue
                        if isinstance(value, MathValue):
                            variables[key] = value
                    except ImportError:
                        pass
                    
                    # Try to import NumberWithUnits to check instance
                    try:
                        from pg.macros.contexts.context_units import NumberWithUnits
                        if isinstance(value, NumberWithUnits):
                            variables[key] = value
                    except ImportError:
                        pass

        return ExecutionResult(
            success=not errors,
            output_text=output_text,
            answers=answers,
            solution_text=solution_text if solution_text else None,
            hint_text=hint_text if hint_text else None,
            errors=errors,
            variables=variables
        )

    def _load_context_macros(self) -> None:
        """Load context-related macros from pg.macros modules."""
        # Use imported modules if available
        self.namespace['LimitedPowers'] = LimitedPowers
        self.namespace['MultiAnswer'] = MultiAnswer
        self.namespace['AnswerHints'] = AnswerHints
        self.namespace['LayoutTable'] = LayoutTable

        # Stub for parser package (minimal implementation)
        class ParserStub:
            """Stub for parser package."""
            class Assignment:
                """Stub for parser::Assignment."""
                @staticmethod
                def Allow(*args):
                    """Call parser::Assignment->Allow to enable assignment operator."""
                    from pg.macros.parsers.parser_assignment import AssignmentParser
                    # Allow() can be called with context as first arg, or just allow=True
                    if len(args) == 0:
                        AssignmentParser.Allow(True, None)
                    elif len(args) == 1:
                        # Could be context or allow flag
                        if isinstance(args[0], bool):
                            AssignmentParser.Allow(args[0], None)
                        else:
                            # Assume it's a context
                            AssignmentParser.Allow(True, args[0])
                    elif len(args) == 2:
                        AssignmentParser.Allow(args[0], args[1])
                    else:
                        AssignmentParser.Allow(True, None)

                @staticmethod
                def Function(*args):
                    """Call parser::Assignment->Function to register function names."""
                    from pg.macros.parsers.parser_assignment import AssignmentParser
                    # Function() takes function names as args
                    if args:
                        AssignmentParser.Function(*args)

        self.namespace['parser'] = ParserStub
        self.namespace['helpLink'] = helpLink

        # Stub for parserFunction - defines named function in context
        self.namespace['parserFunction'] = parserFunction

    def _load_parser_macros(self) -> None:
        """Load parser macros from pg.macros modules."""
        # Use imported parser macros
        self.namespace['PopUp'] = PopUp
        self.namespace['DropDown'] = DropDown
        self.namespace['DropDownTF'] = DropDownTF
        self.namespace['RadioButtons'] = RadioButtons
        self.namespace['RadioMultiAnswer'] = RadioMultiAnswer
        self.namespace['LinearRelation'] = LinearRelation
        self.namespace['DifferenceQuotient'] = DifferenceQuotient
        self.namespace['specialRadical'] = specialRadical
        self.namespace['specialAngle'] = specialAngle

    def _load_statistics_macros(self) -> None:
        """Load statistics functions from pg.macros modules."""
        # Use imported statistics functions
        self.namespace['stats_mean'] = stats_mean
        self.namespace['stats_sd'] = stats_sd
        self.namespace['stats_SX_SXX'] = stats_SX_SXX
        self.namespace['sample_correlation'] = sample_correlation
        self.namespace['linear_regression'] = linear_regression


def create_in_process_sandbox(timeout: int = 30) -> InProcessSandbox:
    """
    Create an in-process sandbox.

    Args:
        timeout: Maximum execution time in seconds

    Returns:
        InProcessSandbox instance
    """
    return InProcessSandbox(timeout=timeout)
