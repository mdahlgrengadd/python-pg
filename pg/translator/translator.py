"""
PG Translator - Coordinate problem file translation pipeline.

Orchestrates:
1. Load .pg file
2. Preprocess (BEGIN_TEXT expansion, etc.)
3. Execute in sandbox
4. Render text (PGML → HTML)
5. Collect answers
6. Check answers (if inputs provided)

Reference: Translator.pm::translate() (lines 679-794)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pg.answer import AnswerResult
from pg.parser import Context

from .error_handler import PGError, format_execution_error, install_error_handlers
from .executor import PGEnvironment, PGExecutor
from .grading import (
    ProblemGrader,
    process_checkbox_radio_input,
    std_problem_grader,
    stringify_answers,
)
from .post_processor import ContentPostProcessor
# Use the structured Pygments/Lark preprocessor by default.
# The legacy regex-based preprocessor remains available via
# pg_translator.LegacyPGPreprocessor if needed.
from .pg_preprocessor_pygment import PGPreprocessor


@dataclass
class ProblemResult:
    """
    Result of translating a PG problem.

    Contains all generated content and answer checking results.
    """

    statement_html: str
    """Problem statement HTML"""

    answer_blanks: dict[str, Any]
    """Answer blank information by name"""

    solution_html: str | None = None
    """Solution HTML (if available)"""

    hint_html: str | None = None
    """Hint HTML (if available)"""

    metadata: dict[str, Any] | None = None
    """Problem metadata"""

    answer_results: dict[str, AnswerResult] | None = None
    """Answer checking results (if inputs provided)"""

    score: float | None = None
    """Overall problem score (if answers checked)"""

    problem_result: dict[str, Any] | None = None
    """Grading result details"""

    problem_state: dict[str, Any] | None = None
    """Problem state (attempts, recorded score, etc.)"""

    header_html: str | None = None
    """Header HTML (CSS, JS, etc.)"""

    errors: list[str] | None = None
    """Execution errors"""

    warnings: list[str] | None = None
    """Warning messages"""


class PGTranslator:
    """
    Translate PG problem files to renderable problems.

    Coordinates the entire pipeline:
    - Preprocessing
    - Safe execution
    - Text rendering
    - Answer checking
    """

    def __init__(
        self,
        preprocessor: PGPreprocessor | None = None,
        executor: PGExecutor | None = None,
        grader: ProblemGrader | None = None,
    ):
        """
        Initialize translator.

        Args:
            preprocessor: PG preprocessor (creates default if None)
            executor: PG executor (creates default if None)
            grader: Problem grader (uses std_problem_grader if None)
        """
        self.preprocessor = preprocessor or PGPreprocessor()
        self.executor = executor or PGExecutor()
        self.grader = grader or std_problem_grader
        self.post_processor = ContentPostProcessor()

        # Macros are now imported via preprocessor - no runtime loading needed

    def translate(
        self,
        pg_file_path: str | Path,
        seed: int,
        inputs: dict[str, Any] | None = None,
        context: Context | None = None,
        problem_state: dict[str, Any] | None = None,
        grader: ProblemGrader | None = None,
    ) -> ProblemResult:
        """
        Translate a PG file to a renderable problem.

        Args:
            pg_file_path: Path to .pg file
            seed: Random seed for problem generation
            inputs: Student answer inputs (for checking)
            context: Mathematical context
            problem_state: Current problem state (optional)
            grader: Custom grader override

        Returns:
            ProblemResult with statement, answers, solutions, etc.
        """
        pg_path = Path(pg_file_path)

        if not pg_path.exists():
            return ProblemResult(
                statement_html="",
                answer_blanks={},
                errors=[f"File not found: {pg_path}"],
            )

        try:
            pg_source = pg_path.read_text(encoding="utf-8")
        except Exception as error:
            error_msg = format_execution_error(error, "", None)
            return ProblemResult(
                statement_html="",
                answer_blanks={},
                errors=[error_msg],
            )

        return self.translate_source(
            pg_source,
            seed,
            inputs=inputs,
            context=context,
            problem_state=problem_state,
            grader=grader,
            filename=str(pg_path),
        )

    def translate_source(
        self,
        pg_source: str,
        seed: int,
        inputs: dict[str, Any] | None = None,
        context: Context | None = None,
        problem_state: dict[str, Any] | None = None,
        grader: ProblemGrader | None = None,
        filename: str | None = None,
    ) -> ProblemResult:
        """
        Translate PG source code directly (without file).

        Args:
            pg_source: PG source code
            seed: Random seed
            inputs: Student answer inputs
            context: Mathematical context
            problem_state: Current problem state
            grader: Problem grader override
            filename: Optional filename for metadata

        Returns:
            ProblemResult
        """
        active_grader = grader or self.grader
        state = (
            dict(problem_state)
            if problem_state is not None
            else {
                "recorded_score": 0,
                "num_of_correct_ans": 0,
                "num_of_incorrect_ans": 0,
            }
        )
        environment: PGEnvironment | None = None
        errors: list[str] = []

        # Auto-wrap bare PG snippets with DOCUMENT/ENDDOCUMENT so the core
        # macros have a valid environment during testing.
        if "DOCUMENT" not in pg_source:
            pg_source = f"DOCUMENT()\n{pg_source}\nENDDOCUMENT()\n"

        try:
            preprocess_result = self.preprocessor.preprocess(pg_source)

            environment = self.executor.execute(
                preprocess_result.code,
                seed=seed,
                context=context,
            )
            install_error_handlers(environment)

            statement_html = environment.render_text()
            solution_html = environment.render_solution()
            hint_html = environment.render_hint()
            header_html = getattr(environment, "render_header", lambda: "")()

            normalized_answers = self._normalize_answers(environment.answers)
            environment.answers = normalized_answers

            answer_results: dict[str, AnswerResult] | None = None
            problem_result_dict: dict[str, Any] | None = None

            if inputs:
                answer_results = self._evaluate_answers(environment, inputs)
                problem_result_dict, state = active_grader(
                    answer_results,
                    state,
                    answers_submitted=True,
                )
                stringify_answers(answer_results)
            else:
                problem_result_dict = {
                    "score": 0,
                    "errors": "",
                    "type": "not_submitted",
                    "msg": "",
                }

            per_answer_average: float | None = None
            if answer_results:
                per_answer_scores = [res.score for res in answer_results.values()]
                if per_answer_scores:
                    per_answer_average = sum(per_answer_scores) / len(per_answer_scores)
                else:
                    per_answer_average = 0.0

            display_mode = getattr(environment, "display_mode", "HTML")
            if self.post_processor.processors:
                statement_html, header_html = self.post_processor.process(
                    statement_html,
                    header_html,
                    display_mode,
                    problem_result_dict,
                )

            # Build answer_blanks from environment.answers
            # environment.answers may contain either:
            # 1. Direct evaluator objects
            # 2. PGML spec dicts with 'evaluator' and 'options' keys
            # 3. Legacy dicts with 'ans_eval' key
            answer_blanks = {}
            for name, entry in environment.answers.items():
                if isinstance(entry, dict) and "evaluator" in entry:
                    # PGML spec format - keep the full spec with options
                    answer_blanks[name] = entry
                elif isinstance(entry, dict) and "ans_eval" in entry:
                    # Legacy format - wrap in simple dict
                    answer_blanks[name] = {"evaluator": entry["ans_eval"]}
                else:
                    # Direct evaluator object
                    answer_blanks[name] = {"evaluator": entry}

            if getattr(environment, "errors", None):
                errors.append(str(environment.errors))

            warnings: list[str] | None = None
            warning_tracker = getattr(environment, "_warning_tracker", None)
            if warning_tracker:
                has_debug = getattr(environment, "view_problem_debugging_info", False)
                warning_text = warning_tracker.get_formatted_warnings(has_debug)
                if warning_text:
                    warnings = [warning_text]

            metadata = {
                "seed": seed,
                "num_answers": len(environment.answers),
                "display_mode": display_mode,
            }
            if filename:
                metadata["source_file"] = filename

            score: float | None = None
            if problem_result_dict:
                grader_type = problem_result_dict.get("type")
                if grader_type and grader_type != "std_problem_grader":
                    score = problem_result_dict.get("score")
                elif per_answer_average is not None:
                    score = per_answer_average
                else:
                    score = problem_result_dict.get("score")
            else:
                score = per_answer_average

            return ProblemResult(
                statement_html=statement_html,
                header_html=header_html,
                answer_blanks=answer_blanks,
                solution_html=solution_html,
                hint_html=hint_html,
                answer_results=answer_results,
                score=score,
                problem_result=problem_result_dict,
                problem_state=state,
                metadata=metadata,
                errors=errors if errors else None,
                warnings=warnings,
            )

        except PGError as error:
            return ProblemResult(
                statement_html="",
                answer_blanks={},
                errors=[str(error)],
            )
        except Exception as error:
            error_msg = format_execution_error(error, pg_source, environment)
            return ProblemResult(
                statement_html="",
                answer_blanks={},
                errors=[error_msg],
            )

    def _cmp_context_flags(self, ans_result: AnswerResult) -> dict[str, Any]:
        """
        Get context flags to set for student answer parsing (Perl cmp_contextFlags equivalent).
        
        Reference: lib/Value/AnswerChecker.pm::cmp_contextFlags (lines 56-82)
        
        Args:
            ans_result: AnswerResult (acts as ans_hash)
            
        Returns:
            Dictionary of flag name -> value to set
        """
        # Get options from metadata (including those from cmp options)
        cmp_options = ans_result.metadata.get('cmp_options', {})

        # Get studentsMustReduceUnions from metadata (if set)
        students_must_reduce_unions = ans_result.metadata.get('studentsMustReduceUnions', False)
        show_union_reduce_warnings = ans_result.metadata.get('showUnionReduceWarnings', False)
        require_paren_match = ans_result.metadata.get('requireParenMatch', False)

        # Get studentsMustReduceFractions from cmp options
        students_must_reduce_fractions = cmp_options.get(
            'studentsMustReduceFractions', False)

        flags = {
            'StringifyAsTeX': 0,  # reset this, just in case
            'no_parameters': 1,  # don't let students enter parameters
            'showExtraParens': 2,  # make student answer painfully unambiguous
            'reduceConstants': 0,  # don't combine student constants
            'reduceConstantFunctions': 0,  # don't reduce constant functions
        }

        # Fraction reduction flags
        # When studentsMustReduceFractions is enabled, disable reduceFractions
        # so we can check if the student's fraction is already reduced
        if students_must_reduce_fractions:
            flags['reduceFractions'] = 0
        else:
            flags['reduceFractions'] = 1

        # Union/Set reduction flags
        if students_must_reduce_unions:
            flags.update({
                'reduceUnions': 0,
                'reduceSets': 0,
                'reduceUnionsForComparison': 1 if show_union_reduce_warnings else 0,
                'reduceSetsForComparison': 1 if show_union_reduce_warnings else 0,
            })
        else:
            flags.update({
                'reduceUnions': 1,
                'reduceSets': 1,
                'reduceUnionsForComparison': 1,
                'reduceSetsForComparison': 1,
            })
        
        # Interval endpoint types
        if not require_paren_match:
            flags['ignoreEndpointTypes'] = 1
        
        return flags
    
    def _set_context_flags(self, context: Any, flags: dict[str, Any]) -> dict[str, Any]:
        """
        Set context flags and return old values for restoration.
        
        Args:
            context: Context object
            flags: Dictionary of flag name -> value to set
            
        Returns:
            Dictionary of old flag values (for restoration)
        """
        if not hasattr(context, 'flags'):
            return {}
        
        old_flags = {}
        for flag_name, flag_value in flags.items():
            old_flags[flag_name] = context.flags.get(flag_name)
            context.flags.set(**{flag_name: flag_value})
        
        return old_flags
    
    def _restore_context_flags(self, context: Any, old_flags: dict[str, Any]) -> None:
        """
        Restore context flags to previous values.
        
        Args:
            context: Context object
            old_flags: Dictionary of old flag values (from _set_context_flags)
        """
        if not hasattr(context, 'flags'):
            return
        
        for flag_name, old_value in old_flags.items():
            if old_value is not None:
                context.flags.set(**{flag_name: old_value})
            # If old_value is None, the flag didn't exist before, so we could remove it
            # But for simplicity, we'll just leave it set
    
    def _cmp_parse(
        self,
        correct_value: Any,
        student_answer: str,
        ans_result: AnswerResult,
        context: Any = None,
    ) -> AnswerResult:
        """
        Parse student answer and generate previews (Perl cmp_parse equivalent).
        
        Reference: lib/Value/AnswerChecker.pm::cmp_parse (lines 122-186)
        
        Args:
            correct_value: The correct MathObject
            student_answer: Raw student input string
            ans_result: AnswerResult to populate
            context: Mathematical context (uses correct_value's context if not provided)
            
        Returns:
            AnswerResult with parsed student_value, student_formula, and previews
        """
        from pg.math.compute import Compute
        from pg.math.formula import Formula
        
        # Get context from correct value if not provided
        if context is None:
            context = getattr(correct_value, 'context', None)
            if context is None:
                from pg.math.context import get_current_context
                context = get_current_context()
        
        # Store original student answer
        ans_result.original_student_answer = student_answer
        ans_result.correct_value = correct_value
        
        # Clear error flags
        ans_result.error_flag = False
        ans_result.error_message = ""
        ans_result.answer_message = ""
        ans_result.preview = ""
        
        # Assume failure initially
        ans_result.score = 0.0
        ans_result.correct = False
        
        # Set context flags for student answer parsing (Perl cmp_contextFlags)
        cmp_flags = self._cmp_context_flags(ans_result)
        old_flags = self._set_context_flags(context, cmp_flags)
        
        # Try to parse student answer
        try:
            # Determine the expected type from correct_value
            correct_type = type(correct_value).__name__
            
            # Parse based on expected type (Perl uses type-specific parsers)
            student_value = None
            student_formula = None
            
            if correct_type == "List":
                # Parse as List
                from pg.math.collections import List
                from pg.math.compute import Compute
                
                # Get the correct List to check element types
                correct_list = correct_value
                expected_elem_type = None
                if hasattr(correct_list, 'elements') and len(correct_list.elements) > 0:
                    # Check what type the first element is
                    first_elem = correct_list.elements[0]
                    expected_elem_type = type(first_elem).__name__
                
                # Try to parse as a list - format: "item1, item2, item3" or "(item1, item2)"
                # First try Compute to see if it gives us a list
                parsed = Compute(student_answer, context)
                
                # If Compute gave us a List, check if element types match
                use_computed_list = False
                if isinstance(parsed, List):
                    # Check if element types match
                    if expected_elem_type and len(parsed.elements) > 0:
                        parsed_elem_type = type(parsed.elements[0]).__name__
                        if parsed_elem_type == expected_elem_type:
                            use_computed_list = True
                    elif not expected_elem_type:
                        # No expected type specified, use computed list
                        use_computed_list = True
                
                if use_computed_list:
                    student_value = parsed
                    student_formula = parsed
                else:
                    # If Compute didn't give us a List or types don't match, parse manually
                    # Try to parse manually as comma-separated list
                    # Remove outer brackets if present
                    cleaned = student_answer.strip()
                    
                    # If Compute returned a Formula that might represent a tuple/list, try evaluating it
                    # This handles cases like Compute("((1, 0), (-1, 0))") which returns a Formula
                    from pg.math.formula import Formula
                    if isinstance(parsed, Formula):
                        try:
                            # Check if it's a constant formula
                            if hasattr(parsed, 'isConstant') and parsed.isConstant():
                                eval_result = parsed.eval()
                                # If eval returns a tuple/list, use that for parsing
                                if isinstance(eval_result, (list, tuple)):
                                    from pg.math.value import MathValue
                                    elements = [MathValue.from_python(el) for el in eval_result]
                                    # Check if we need to convert elements to the expected type
                                    if expected_elem_type:
                                        converted_elements = []
                                        for el in elements:
                                            if type(el).__name__ != expected_elem_type:
                                                # Try to convert to expected type
                                                if expected_elem_type == "Point" and isinstance(el, (list, tuple)):
                                                    from pg.math.geometric import Point
                                                    if len(el) >= 2:
                                                        converted_elements.append(Point(el[0], el[1]))
                                                    else:
                                                        converted_elements.append(el)
                                                else:
                                                    converted_elements.append(el)
                                            else:
                                                converted_elements.append(el)
                                        elements = converted_elements
                                    student_value = List(elements)
                                    student_formula = student_value
                                    # Skip manual parsing since we got a result from eval
                                    cleaned = None
                        except Exception:
                            # If eval fails, continue with manual parsing
                            pass
                    
                    if cleaned is not None:
                        if cleaned.startswith('[') and cleaned.endswith(']'):
                            cleaned = cleaned[1:-1].strip()
                        
                        # Split by comma (handling nested parentheses)
                        elements = []
                        current = ""
                        depth = 0
                        for char in cleaned:
                            if char in '([{':
                                depth += 1
                                current += char
                            elif char in ')]}':
                                depth -= 1
                                current += char
                            elif char == ',' and depth == 0:
                                if current.strip():
                                    elem_str = current.strip()
                                    # Try to parse element according to expected type
                                    elem_parsed = None
                                    try:
                                        if expected_elem_type == "Point":
                                            # Parse as Point - format: "(x, y)" or "((x, y))"
                                            from pg.math.geometric import Point
                                            # Remove extra parentheses if present
                                            cleaned_elem = elem_str.strip()
                                            if cleaned_elem.startswith('((') and cleaned_elem.endswith('))'):
                                                cleaned_elem = cleaned_elem[1:-1].strip()
                                            elem_parsed = Compute(cleaned_elem, context)
                                            # If Compute didn't give us a Point, try creating one
                                            if not isinstance(elem_parsed, Point):
                                                # Try to extract coordinates
                                                import re
                                                # Match (x, y) pattern
                                                match = re.match(r'\(([^,]+),\s*([^)]+)\)', cleaned_elem)
                                                if match:
                                                    x_str, y_str = match.groups()
                                                    x = Compute(x_str.strip(), context)
                                                    y = Compute(y_str.strip(), context)
                                                    elem_parsed = Point(x, y)
                                                else:
                                                    elem_parsed = Compute(elem_str, context)
                                        else:
                                            # For other types, just use Compute
                                            elem_parsed = Compute(elem_str, context)
                                    except Exception:
                                        # If parsing fails, try Compute without type-specific handling
                                        try:
                                            elem_parsed = Compute(elem_str, context)
                                        except Exception:
                                            elem_parsed = None
                                    
                                    if elem_parsed is not None:
                                        elements.append(elem_parsed)
                                current = ""
                            else:
                                current += char
                        if current.strip():
                            elem_str = current.strip()
                            # Same parsing logic as above
                            elem_parsed = None
                            try:
                                if expected_elem_type == "Point":
                                    from pg.math.geometric import Point
                                    cleaned_elem = elem_str.strip()
                                    if cleaned_elem.startswith('((') and cleaned_elem.endswith('))'):
                                        cleaned_elem = cleaned_elem[1:-1].strip()
                                    elem_parsed = Compute(cleaned_elem, context)
                                    if not isinstance(elem_parsed, Point):
                                        import re
                                        match = re.match(r'\(([^,]+),\s*([^)]+)\)', cleaned_elem)
                                        if match:
                                            x_str, y_str = match.groups()
                                            x = Compute(x_str.strip(), context)
                                            y = Compute(y_str.strip(), context)
                                            elem_parsed = Point(x, y)
                                        else:
                                            elem_parsed = Compute(elem_str, context)
                                else:
                                    elem_parsed = Compute(elem_str, context)
                            except Exception:
                                try:
                                    elem_parsed = Compute(elem_str, context)
                                except Exception:
                                    elem_parsed = None
                            
                            if elem_parsed is not None:
                                elements.append(elem_parsed)
                        
                        if elements:
                            student_value = List(elements)
                            student_formula = student_value
                        elif isinstance(parsed, List):
                            # Fallback to Compute result if it's a List (even if types don't match)
                            student_value = parsed
                            student_formula = parsed
                        else:
                            # Last resort: try to convert parsed result to List if it's a Python list/tuple
                            # This handles cases where Compute returns a Formula that evaluates to a tuple
                            if isinstance(parsed, (list, tuple)):
                                # Convert Python list/tuple to List MathObject
                                from pg.math.value import MathValue
                                elements = [MathValue.from_python(el) for el in parsed]
                                student_value = List(elements)
                                student_formula = student_value
                            else:
                                # Use Compute result as-is
                                student_formula = parsed
                                student_value = parsed
            
            elif correct_type == "Matrix":
                # Parse as Matrix
                from pg.math.geometric import Matrix
                import ast
                
                try:
                    # Try parsing as Python list literal
                    parsed = ast.literal_eval(student_answer)
                    if isinstance(parsed, list):
                        student_value = Matrix(parsed)
                        student_formula = student_value
                    else:
                        raise ValueError("Not a matrix format")
                except (ValueError, SyntaxError):
                    # Fallback to Compute
                    parsed = Compute(student_answer, context)
                    if isinstance(parsed, Matrix):
                        student_value = parsed
                        student_formula = parsed
                    else:
                        student_formula = parsed
                        student_value = parsed
            
            elif correct_type == "String":
                # Parse as String literal
                from pg.math.collections import String
                
                # String should be the literal value (no quotes needed in PG)
                student_value = String(student_answer)
                student_formula = student_value
            
            else:
                # For other types (Formula, Real, Vector, Point, etc.), use Compute
                # Compute() will now handle assignment parsing if context has assignments enabled
                student_formula = Compute(student_answer, context)
                
                # If it's a constant formula, evaluate it
                is_constant = False
                if hasattr(student_formula, 'isConstant'):
                    is_constant = student_formula.isConstant()
                elif hasattr(student_formula, 'variables'):
                    is_constant = len(student_formula.variables) == 0
                
                if is_constant:
                    try:
                        # Evaluate constant formula to get numeric value
                        student_value = student_formula.eval()
                    except Exception:
                        # If evaluation fails, use formula as value
                        student_value = student_formula
                else:
                    # Non-constant formula - use formula as value
                    student_value = student_formula
            
            ans_result.student_formula = student_formula
            ans_result.student_value = student_value
            
            # Generate preview strings
            if hasattr(student_formula, 'to_tex'):
                ans_result.preview = student_formula.to_tex()
            elif hasattr(student_formula, 'to_string'):
                ans_result.preview = student_formula.to_string()
            else:
                ans_result.preview = str(student_formula)
            
            # Set student_answer to parsed/normalized form
            if hasattr(student_formula, 'to_string'):
                ans_result.student_answer = student_formula.to_string()
            else:
                ans_result.student_answer = str(student_formula)
            
            # Mark as student value (for potential special handling)
            if hasattr(ans_result.student_value, '__dict__'):
                ans_result.student_value.__dict__['isStudent'] = True
            
        except Exception as e:
            # Parsing failed - set error
            ans_result.student_answer = student_answer  # Use raw input
            ans_result.student_value = None
            ans_result.student_formula = None
            ans_result.set_error(f"Could not parse answer: {str(e)}")
        finally:
            # Restore context flags (Perl: contextSet($context, %{$flags}))
            self._restore_context_flags(context, old_flags)
        
        return ans_result
    
    def _cmp_equal(
        self,
        correct_value: Any,
        student_value: Any,
        ans_result: AnswerResult,
        custom_checker: Any = None,
    ) -> AnswerResult:
        """
        Check if parsed student answer equals correct answer (Perl cmp_equal equivalent).
        
        Reference: lib/Value/AnswerChecker.pm::cmp_equal (lines 233-255)
        
        Args:
            correct_value: The correct MathObject
            student_value: Parsed student MathObject
            ans_result: AnswerResult to populate
            custom_checker: Custom checker function (if provided)
            
        Returns:
            AnswerResult with comparison result
        """
        # Check type matching first (unless custom checker handles it)
        if custom_checker is None:
            if not self._type_match(correct_value, student_value, ans_result):
                # Type mismatch - already handled in _type_match
                return ans_result
        
        # Types match - perform comparison
        try:
            equal = self._cmp_compare(
                correct_value, student_value, ans_result,
                custom_checker=custom_checker
            )
            if equal is not None and equal:
                ans_result.score = 1.0
                ans_result.correct = True
                if not ans_result.answer_message:
                    ans_result.answer_message = "Correct!"
            elif equal is False:
                ans_result.score = 0.0
                ans_result.correct = False
                if not ans_result.answer_message:
                    ans_result.answer_message = "Incorrect."
        except Exception as e:
            ans_result.set_error(f"Error during comparison: {str(e)}")
        
        return ans_result
    
    def _type_match(
        self,
        correct_value: Any,
        student_value: Any,
        ans_result: AnswerResult,
    ) -> bool:
        """
        Check if types are compatible for equality check (Perl typeMatch equivalent).
        
        Reference: lib/Value/AnswerChecker.pm::typeMatch (lines 295-300)
        
        Args:
            correct_value: The correct MathObject
            student_value: Parsed student MathObject
            ans_result: AnswerResult (for error messages)
            
        Returns:
            True if types match, False otherwise
        """
        # If student_value is not a MathObject, types don't match
        if student_value is None:
            return False
        
        # Get types
        correct_type = type(correct_value).__name__
        student_type = type(student_value).__name__
        
        # Check if types match (excluding Formula types)
        if correct_type == student_type:
            # Check if student is a Formula (special handling)
            if hasattr(student_value, 'isFormula') and student_value.isFormula():
                # Formulas can match if they evaluate to the same type
                # This is handled in cmp_compare
                return True
            return True
        
        # Type mismatch - set error message
        ans_result.typeError = True
        ans_result.answer_message = (
            f"Your answer isn't {correct_type.lower()}\n"
            f"(it looks like {student_type.lower()})"
        )
        ans_result.error_message = ans_result.answer_message
        ans_result.score = 0.0
        ans_result.correct = False
        
        return False
    
    def _call_checker_method(self, checker: Any, student_answer: str, ans_hash: AnswerResult) -> bool:
        """Helper to call checker.check() method."""
        result = checker.check(student_answer)
        if isinstance(result, dict):
            if "message" in result:
                ans_hash.answer_message = result["message"]
            score = result.get("score", 0.0)
            ans_hash.score = score
            ans_hash.correct = score >= 1.0
            return score >= 1.0
        return False
    
    def _call_checker_lambda(self, checker: Any, student_answer: str, ans_hash: AnswerResult) -> bool:
        """Helper to call checker as a lambda function."""
        result = checker(student_answer)
        if isinstance(result, dict):
            if "message" in result:
                ans_hash.answer_message = result["message"]
            score = result.get("score", 0.0)
            ans_hash.score = score
            ans_hash.correct = score >= 1.0
            return score >= 1.0
        return False
    
    def _cmp_compare(
        self,
        correct_value: Any,
        student_value: Any,
        ans_result: AnswerResult,
        nth: str = "",
        custom_checker: Any = None,
    ) -> bool | None:
        """
        Perform the comparison using custom checker or operator (Perl cmp_compare equivalent).

        Reference: lib/Value/AnswerChecker.pm::cmp_compare (lines 266-288)

        Args:
            correct_value: The correct MathObject
            student_value: Parsed student MathObject
            ans_result: AnswerResult (acts as ans_hash)
            nth: Which answer in MultiAnswer (for error messages)
            custom_checker: Custom checker function (if provided)

        Returns:
            True if equal, False if not equal, None if error
        """
        # If custom checker provided, use it
        if custom_checker is not None and callable(custom_checker):
            try:
                # Perl signature: ($correct, $student, $ans_hash, $nth, @extra)
                # Python equivalent: (correct, student, ans_result, nth, *extra)
                result = custom_checker(correct_value, student_value, ans_result, nth)
                
                # Handle different return types
                if isinstance(result, (list, tuple)):
                    # Multiple return values (score, message, etc.)
                    if len(result) > 0:
                        score = result[0]
                        if isinstance(score, (int, float)):
                            return bool(score >= 1.0)
                        return bool(score)
                    return None
                elif isinstance(result, bool):
                    return result
                elif isinstance(result, (int, float)):
                    return bool(result >= 1.0)
                else:
                    return None
                    
            except Exception as e:
                # Error in custom checker
                error_msg = (
                    f"<I>An error occurred while checking your{nth} answer:</I>\n"
                    f'<DIV STYLE="margin-left:1em">{str(e)}</DIV>'
                )
                ans_result.set_error(error_msg)
                return None
        
        # No custom checker - use overloaded == operator
        try:
            # Use Python's == operator (which should call MathObject's __eq__)
            # Assignment formulas will be handled by Formula.compare() method
            return bool(correct_value == student_value)
        except Exception as e:
            ans_result.set_error(f"Error during comparison: {str(e)}")
            return None

    def _evaluate_answers(
        self,
        environment: PGEnvironment,
        raw_inputs: dict[str, Any],
    ) -> dict[str, AnswerResult]:
        """
        Evaluate student answers, handling checkbox/radio inputs and MultiAnswer groups.
        
        Now uses Perl-equivalent cmp_parse → cmp_equal → cmp_compare flow.
        """
        processed_inputs = {
            name: process_checkbox_radio_input(value)
            for name, value in raw_inputs.items()
        }

        answer_results: dict[str, AnswerResult] = {}
        evaluator_groups: dict[int, list[tuple[str, Any]]] = {}
        evaluator_map: dict[int, Any] = {}

        for name, student_answer in processed_inputs.items():
            if name not in environment.answers:
                continue

            ans_entry = environment.answers[name]

            # Handle different answer entry formats
            if isinstance(ans_entry, dict) and "ans_eval" in ans_entry:
                # Legacy format with explicit ans_eval key
                evaluator = ans_entry["ans_eval"]
                cmp_options = {}
            elif isinstance(ans_entry, dict) and "evaluator" in ans_entry:
                # PGML spec format with evaluator and options
                evaluator = ans_entry["evaluator"]
                cmp_options = ans_entry.get("options", {})
            else:
                # Direct evaluator object
                evaluator = ans_entry
                cmp_options = {}

            eval_id = id(evaluator)
            evaluator_groups.setdefault(eval_id, []).append((name, student_answer, cmp_options))
            evaluator_map[eval_id] = evaluator

        for eval_id, group_items in evaluator_groups.items():
            evaluator = evaluator_map[eval_id]

            if len(group_items) > 1 and hasattr(evaluator, "cmp"):
                # Get options from the first item in the group (all should have the same)
                cmp_options = group_items[0][2] if len(group_items[0]) > 2 else {}
                checker = evaluator.cmp(**cmp_options)
                if hasattr(checker, "check"):
                    student_answers = [ans for _, ans, _ in group_items]
                    check_result = checker.check(*student_answers)

                    if "results" in check_result and isinstance(check_result["results"], list):
                        answers = getattr(evaluator, "answers", [])
                        for index, (name, student_answer, _cmp_opts) in enumerate(group_items):
                            individual_score = (
                                check_result["results"][index]
                                if index < len(check_result["results"])
                                else 0.0
                            )
                            correct_answer = ""
                            if isinstance(answers, list) and index < len(answers):
                                correct_answer = str(answers[index])

                            answer_results[name] = AnswerResult(
                                score=individual_score,
                                correct=individual_score >= 1.0,
                                student_answer=student_answer,
                                answer_message=check_result.get("message", ""),
                                correct_answer=correct_answer,
                            )
                    else:
                        for name, student_answer, _cmp_opts in group_items:
                            answer_results[name] = AnswerResult(
                                score=check_result.get("score", 0.0),
                                correct=check_result.get("correct", False),
                                student_answer=student_answer,
                                answer_message=check_result.get("message", ""),
                                correct_answer=str(evaluator),
                            )
                elif hasattr(checker, "evaluate"):
                    # MultiAnswer uses .evaluate() instead of .check()
                    # Call evaluate with all student answers
                    student_answers = [ans for _, ans, _ in group_items]
                    try:
                        eval_result = checker.evaluate(*student_answers)
                    except Exception as e:
                        # Log error and create error results
                        error_msg = f"Error evaluating MultiAnswer: {str(e)}"
                        for name, student_answer, _cmp_opts in group_items:
                            answer_results[name] = AnswerResult(
                                score=0.0,
                                correct=False,
                                student_answer=student_answer,
                                answer_message=error_msg,
                                correct_answer=str(evaluator),
                            )
                        continue

                    # Handle None result (evaluator not implemented properly)
                    if eval_result is None:
                        for name, student_answer, _cmp_opts in group_items:
                            answer_results[name] = AnswerResult(
                                score=0.0,
                                correct=False,
                                student_answer=student_answer,
                                answer_message="MultiAnswer evaluator returned None",
                                correct_answer=str(evaluator),
                            )
                        continue

                    # If evaluate returns an AnswerResult, extract the score and info
                    if hasattr(eval_result, "score"):
                        # Single result for all answers
                        score = eval_result.score
                        message = getattr(eval_result, "answer_message", "")
                        for name, student_answer, _cmp_opts in group_items:
                            answer_results[name] = AnswerResult(
                                score=score,
                                correct=score >= 1.0,
                                student_answer=student_answer,
                                answer_message=message,
                                correct_answer=str(evaluator),
                            )
                    elif isinstance(eval_result, dict):
                        # Dictionary with individual results
                        if "results" in eval_result and isinstance(eval_result["results"], list):
                            answers = getattr(evaluator, "answers", [])
                            for index, (name, student_answer, _cmp_opts) in enumerate(group_items):
                                individual_score = (
                                    eval_result["results"][index]
                                    if index < len(eval_result["results"])
                                    else 0.0
                                )
                                correct_answer = ""
                                if isinstance(answers, list) and index < len(answers):
                                    correct_answer = str(answers[index])
                                
                                answer_results[name] = AnswerResult(
                                    score=individual_score,
                                    correct=individual_score >= 1.0,
                                    student_answer=student_answer,
                                    answer_message=eval_result.get("message", ""),
                                    correct_answer=correct_answer,
                                )
                        else:
                            # Single score for all
                            score = eval_result.get("score", 0.0)
                            for name, student_answer, _cmp_opts in group_items:
                                answer_results[name] = AnswerResult(
                                    score=score,
                                    correct=score >= 1.0,
                                    student_answer=student_answer,
                                    answer_message=eval_result.get("message", ""),
                                    correct_answer=str(evaluator),
                                )
                continue

            for name, student_answer, cmp_options in group_items:
                # Check for cmp() first (MathObjects with Perl-equivalent flow)
                # This must come before compare() check since many MathObjects have both
                if hasattr(evaluator, "cmp"):
                    # MathObject with cmp() - use Perl-equivalent flow
                    checker = evaluator.cmp(**cmp_options)

                    # Create initial AnswerResult
                    ans_result = AnswerResult(
                        original_student_answer=student_answer,
                        ans_label=name,
                        type=f"Value ({type(evaluator).__name__})",
                        correct_answer=str(evaluator) if hasattr(evaluator, "__str__") else "",
                        metadata={'cmp_options': cmp_options},
                    )
                    
                    # Step 1: Parse student answer (cmp_parse equivalent)
                    ans_result = self._cmp_parse(evaluator, student_answer, ans_result)
                    
                    # If parsing failed with an error, use the error result
                    if ans_result.error_flag:
                        answer_results[name] = ans_result
                        continue
                    
                    # If student_value is None but no error, try to proceed anyway
                    # (some types might not set student_value even after successful parsing)
                    if ans_result.student_value is None:
                        # Try to use student_formula as student_value
                        if ans_result.student_formula is not None:
                            ans_result.student_value = ans_result.student_formula
                        else:
                            # Fall back to string comparison
                            ans_result.student_value = student_answer
                    
                    # Step 2: Get custom checker if available
                    # Note: For now, we'll use the checker's .check() method directly
                    # The Perl signature alignment is handled in _cmp_compare
                    custom_checker = None
                    if hasattr(checker, "check"):
                        # Checker has a .check() method
                        # We'll call it in _cmp_compare with the parsed student value
                        # Store the checker for use in _cmp_compare
                        custom_checker = lambda correct, student, ans_hash, nth: self._call_checker_method(
                            checker, student_answer, ans_hash
                        )
                    elif callable(checker):
                        # PopUp/DropDown/RadioButtons return a callable lambda
                        # Store the checker for use in _cmp_compare
                        custom_checker = lambda correct, student, ans_hash, nth: self._call_checker_lambda(
                            checker, student_answer, ans_hash
                        )
                    
                    # Step 3: Compare (cmp_equal → cmp_compare)
                    # Pass custom_checker to cmp_equal
                    ans_result = self._cmp_equal(
                        evaluator,
                        ans_result.student_value,
                        ans_result,
                        custom_checker=custom_checker,
                    )
                    
                    answer_results[name] = ans_result
                elif hasattr(evaluator, "check") and not hasattr(evaluator, "cmp"):
                    check_result = evaluator.check(student_answer)
                    answer_results[name] = AnswerResult(
                        score=check_result.get("score", 0.0),
                        correct=check_result.get("correct", False),
                        student_answer=student_answer,
                        answer_message=check_result.get("message", ""),
                        correct_answer=check_result.get(
                            "correct_answer",
                            str(evaluator),
                        ),
                    )
                elif hasattr(evaluator, "compare") and callable(evaluator.compare):
                    # MathObject types (List, Point, Vector, Matrix, etc.) use .compare() method
                    try:
                        evaluator_type = type(evaluator).__name__
                        evaluator_str = str(evaluator).strip()
                        student_str = student_answer.strip()
                        
                        # Try to parse student answer and create MathObject for proper comparison
                        student_obj = None
                        is_correct = False
                        
                        if evaluator_type == "Matrix":
                            # Matrix format: [[1, 2, 3], [4, 5, 6]]
                            import ast
                            try:
                                parsed = ast.literal_eval(student_str)
                                from pg.math.geometric import Matrix
                                student_obj = Matrix(parsed)
                                # compare() returns bool (True if equal)
                                is_correct = evaluator.compare(student_obj)
                            except (ValueError, SyntaxError, TypeError) as e:
                                # Fall back to string comparison
                                is_correct = (evaluator_str == student_str)
                        elif evaluator_type == "List":
                            # List format: "a, b, c" (without brackets)
                            # Remove brackets from evaluator string representation
                            eval_str = evaluator_str
                            if eval_str.startswith('[') and eval_str.endswith(']'):
                                eval_str = eval_str[1:-1].strip()
                            is_correct = (eval_str == student_str)
                        else:
                            # For other types (Point, Vector, etc.), try string comparison
                            # TODO: Add proper parsing for Point/Vector types
                            is_correct = (evaluator_str == student_str)
                        
                        answer_results[name] = AnswerResult(
                            score=1.0 if is_correct else 0.0,
                            correct=is_correct,
                            student_answer=student_answer,
                            answer_message="" if is_correct else "Incorrect",
                            correct_answer=evaluator_str,
                        )
                    except Exception as e:
                        answer_results[name] = AnswerResult(
                            score=0.0,
                            correct=False,
                            student_answer=student_answer,
                            answer_message=f"Error comparing answer: {str(e)}",
                            correct_answer=str(evaluator),
                        )
                elif hasattr(evaluator, "evaluate"):
                    result = evaluator.evaluate(student_answer)
                    answer_results[name] = result

        return answer_results

    def _normalize_answers(
        self,
        answers: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Normalize answer registry, applying aliases from ANS label arguments.
        """
        normalized: dict[str, Any] = {}
        last_key: str | None = None
        pending_alias: str | None = None

        for name, entry in answers.items():
            evaluator = (
                entry["ans_eval"]
                if isinstance(entry, dict) and "ans_eval" in entry
                else entry
            )

            if isinstance(evaluator, str):
                alias = evaluator.strip()
                if not alias:
                    continue
                if last_key is not None and last_key in normalized:
                    normalized[alias] = normalized.pop(last_key)
                    last_key = alias
                else:
                    pending_alias = alias
                continue

            target_name = pending_alias or name
            pending_alias = None
            normalized[target_name] = entry
            last_key = target_name

        return normalized
