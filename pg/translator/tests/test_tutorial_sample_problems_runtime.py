"""
Runtime testing for tutorial sample problems.

This test suite validates that answer checking/grading works correctly for
tutorial sample problems by:
1. Generating each problem with a fixed seed
2. Extracting the correct answers from MathObject evaluators
3. Re-running the problem with correct answers as inputs
4. Verifying that the submission scores 100% (full credit)

This complements the compilation tester which only validates that problems
render without errors. The runtime tester ensures the answer checking logic
actually works end-to-end.

Usage:
    # Run all runtime tests
    pytest pg/translator/tests/test_tutorial_sample_problems_runtime.py -v

    # Run by category
    pytest pg/translator/tests/test_tutorial_sample_problems_runtime.py -v -k "Algebra"

    # Run single problem
    pytest pg/translator/tests/test_tutorial_sample_problems_runtime.py -v -k "ExpandedPolynomial"

    # Run aggregate test with detailed results
    pytest pg/translator/tests/test_tutorial_sample_problems_runtime.py::TestTutorialSampleProblemsRuntime::test_all_correct_answers -v

    # Run with short traceback for debugging
    pytest pg/translator/tests/test_tutorial_sample_problems_runtime.py -v --tb=short
"""

import pytest
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Any

# Add packages to path
_pg_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_pg_root / "packages"))

from pg.translator import PGTranslator
from .answer_extraction import extract_correct_answers, get_extractable_blanks


class TestTutorialSampleProblemsRuntime:
    """Runtime answer checking tests for tutorial sample problems."""

    @staticmethod
    def get_all_problem_files() -> List[Path]:
        """
        Discover all .pg files in tutorial/sample-problems directory.

        Returns:
            List of Path objects for all .pg problem files, sorted alphabetically.
        """
        tutorial_dir = _pg_root / "tutorial" / "sample-problems"

        if not tutorial_dir.exists():
            pytest.skip(f"tutorial/sample-problems directory not found at {tutorial_dir}")

        problem_files = sorted(tutorial_dir.rglob("*.pg"))

        if not problem_files:
            pytest.skip("No .pg files found in tutorial/sample-problems")

        return problem_files

    @staticmethod
    def get_category_and_name(problem_path: Path) -> Tuple[str, str]:
        """
        Extract category and problem name from problem path.

        Args:
            problem_path: Path to the .pg file

        Returns:
            Tuple of (category, problem_name)
        """
        parts = problem_path.relative_to(problem_path.parent.parent.parent.parent).parts
        if len(parts) >= 3:
            category = parts[2]
            name = problem_path.stem
            return category, name
        return problem_path.parent.name, problem_path.stem

    @pytest.mark.parametrize(
        "problem_path",
        get_all_problem_files(),
        ids=lambda p: str(p.relative_to(_pg_root / "tutorial" / "sample-problems")).replace("\\", "/")
    )
    def test_correct_answer_scores_full_credit(self, problem_path):
        """
        Test that submitting the correct answer yields a perfect score.

        This is the main runtime test. For each problem:
        1. Generate the problem
        2. Extract the correct answer from the evaluator
        3. Submit the correct answer
        4. Verify score is 1.0 (100%)

        If a problem has no answer blanks, the test is skipped.
        If answer extraction fails, the test is skipped with a note.
        """
        translator = PGTranslator()
        seed = 1234
        category, name = self.get_category_and_name(problem_path)
        test_id = f"{category}/{name}"

        # Step 1: Generate problem and extract correct answers
        result = translator.translate(str(problem_path), seed=seed)

        # Skip if compilation failed
        if result.errors:
            pytest.skip(f"Problem has compilation errors: {result.errors[0]}")

        # Skip if no answer blanks
        if not result.answer_blanks:
            pytest.skip("Problem has no answer blanks")

        # Step 2: Extract correct answers
        correct_answers = extract_correct_answers(result)

        if not correct_answers:
            # Check which blanks couldn't be extracted
            extractable = get_extractable_blanks(result)
            unextractable = [name for name, can_extract in extractable.items() if not can_extract]
            pytest.skip(f"Could not extract answer(s) for blank(s): {', '.join(unextractable)}")

        # Step 3: Re-run with correct answers
        check_result = translator.translate(
            str(problem_path),
            seed=seed,
            inputs=correct_answers
        )

        # Step 4: Verify perfect score
        if check_result.errors:
            pytest.fail(
                f"Answer checking failed with error: {check_result.errors[0]}\n"
                f"Submitted answers: {correct_answers}"
            )

        if check_result.answer_results is None:
            pytest.fail(
                f"Answer checking did not return results\n"
                f"Submitted answers: {correct_answers}"
            )

        # Check overall score
        if check_result.score is None:
            pytest.fail("No overall score returned from answer checking")

        if check_result.score < 1.0:
            # Build detailed failure message
            details = []

            if check_result.answer_results:
                for blank_name, answer_result in check_result.answer_results.items():
                    if answer_result.score < 1.0:
                        details.append(
                            f"  {blank_name}:\n"
                            f"    Score: {answer_result.score}\n"
                            f"    Submitted: {correct_answers.get(blank_name, '?')}\n"
                            f"    Expected: {answer_result.correct_answer}\n"
                            f"    Feedback: {answer_result.answer_message}"
                        )
            else:
                # No answer results returned - answer checker may not have run properly
                details.append(
                    f"  No answer results returned from checker\n"
                    f"  Submitted answers: {correct_answers}"
                )

            pytest.fail(
                f"Correct answer(s) did not score full credit\n"
                f"Overall score: {check_result.score} (expected 1.0)\n"
                f"Details:\n" + "\n".join(details) if details else "No details available"
            )

    def test_all_correct_answers(self):
        """
        Aggregate test for all problems' answer checking.

        This test runs all problems and provides a comprehensive summary
        of which problems pass/skip/fail answer checking.
        """
        problem_files = self.get_all_problem_files()

        if not problem_files:
            pytest.skip("No problem files found")

        translator = PGTranslator()
        seed = 1234
        results = []

        for problem_path in problem_files:
            category, name = self.get_category_and_name(problem_path)
            test_id = f"{category}/{name}"

            try:
                # Generate problem
                result = translator.translate(str(problem_path), seed=seed)

                # Check for compilation errors
                if result.errors:
                    results.append((test_id, "SKIP", f"Compilation error: {result.errors[0][:50]}"))
                    continue

                # Skip if no answer blanks
                if not result.answer_blanks:
                    results.append((test_id, "SKIP", "No answer blanks"))
                    continue

                # Extract correct answers
                correct_answers = extract_correct_answers(result)

                if not correct_answers:
                    extractable = get_extractable_blanks(result)
                    unextractable = [n for n, can_extract in extractable.items() if not can_extract]
                    results.append((test_id, "SKIP", f"Cannot extract answers: {', '.join(unextractable[:2])}"))
                    continue

                # Check answers
                check_result = translator.translate(
                    str(problem_path),
                    seed=seed,
                    inputs=correct_answers
                )

                # Verify score
                if check_result.score is None or check_result.score < 1.0:
                    score_str = f"{check_result.score:.1f}" if check_result.score is not None else "None"
                    results.append((test_id, "FAIL", f"Score {score_str} (expected 1.0)"))
                else:
                    results.append((test_id, "PASS", None))

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)[:40]}"
                results.append((test_id, "ERROR", error_msg))

        # Print summary
        print(f"\n\n{'='*80}")
        print(f"TUTORIAL SAMPLE PROBLEMS - RUNTIME ANSWER CHECKING SUMMARY")
        print(f"{'='*80}\n")

        passed = len([r for r in results if r[1] == "PASS"])
        skipped = len([r for r in results if r[1] == "SKIP"])
        failed = len([r for r in results if r[1] in ("FAIL", "ERROR")])
        total = len(results)

        # Group by status
        by_status = {"PASS": [], "SKIP": [], "FAIL": [], "ERROR": []}
        for test_id, status, reason in results:
            by_status[status].append((test_id, reason))

        # Print results grouped by status
        if by_status["PASS"]:
            print(f"PASSED ({len(by_status['PASS'])}):")
            for test_id, _ in sorted(by_status["PASS"]):
                print(f"  ✓ {test_id}")
            print()

        if by_status["SKIP"]:
            print(f"SKIPPED ({len(by_status['SKIP'])}):")
            for test_id, reason in sorted(by_status["SKIP"]):
                print(f"  ⊘ {test_id}")
                if reason:
                    print(f"      {reason}")
            print()

        if by_status["FAIL"]:
            print(f"FAILED ({len(by_status['FAIL'])}):")
            for test_id, reason in sorted(by_status["FAIL"]):
                print(f"  ✗ {test_id}")
                if reason:
                    print(f"      {reason}")
            print()

        if by_status["ERROR"]:
            print(f"ERRORS ({len(by_status['ERROR'])}):")
            for test_id, reason in sorted(by_status["ERROR"]):
                print(f"  ✗ {test_id}")
                if reason:
                    print(f"      {reason}")
            print()

        # Overall summary
        print(f"{'='*80}")
        print(f"Results: {passed} passed, {skipped} skipped, {failed} failed/error out of {total} total")
        if passed + skipped + failed + len(by_status["ERROR"]) > 0:
            success_rate = passed / (total - skipped) * 100 if total > skipped else 0
            print(f"Success rate (non-skipped): {success_rate:.1f}%")
        print(f"{'='*80}\n")

        # Fail if any problems actually failed (skip doesn't fail the test)
        if failed > 0 or len(by_status["ERROR"]) > 0:
            failure_reasons = []
            for test_id, reason in by_status["FAIL"] + by_status["ERROR"]:
                failure_reasons.append(f"{test_id}: {reason}")
            pytest.fail(
                f"{failed + len(by_status['ERROR'])} problem(s) failed runtime answer checking:\n" +
                "\n".join(failure_reasons[:10])
            )


# Fixtures for advanced testing scenarios

@pytest.fixture
def translator():
    """Provide a PGTranslator instance for tests."""
    return PGTranslator()


@pytest.fixture
def problem_files():
    """Provide list of all problem files."""
    return TestTutorialSampleProblemsRuntime.get_all_problem_files()


@pytest.fixture
def runtime_test_stats(problem_files, translator):
    """
    Generate statistics about runtime testing coverage.

    Shows which problems support runtime testing vs. which have unsupported answer types.
    """
    stats = {
        "total": len(problem_files),
        "extractable": 0,
        "not_extractable": 0,
        "no_answers": 0,
        "compilation_errors": 0,
        "by_category": {},
    }

    for problem_path in problem_files:
        category, _ = TestTutorialSampleProblemsRuntime.get_category_and_name(problem_path)
        if category not in stats["by_category"]:
            stats["by_category"][category] = {"total": 0, "extractable": 0, "skipped": 0}
        stats["by_category"][category]["total"] += 1

        try:
            result = translator.translate(str(problem_path), seed=1234)

            if result.errors:
                stats["compilation_errors"] += 1
                stats["by_category"][category]["skipped"] += 1
            elif not result.answer_blanks:
                stats["no_answers"] += 1
                stats["by_category"][category]["skipped"] += 1
            else:
                correct_answers = extract_correct_answers(result)
                if correct_answers:
                    stats["extractable"] += 1
                    stats["by_category"][category]["extractable"] += 1
                else:
                    stats["not_extractable"] += 1
                    stats["by_category"][category]["skipped"] += 1
        except Exception:
            stats["compilation_errors"] += 1
            stats["by_category"][category]["skipped"] += 1

    return stats


def test_runtime_coverage_stats(runtime_test_stats):
    """Report statistics about runtime testing coverage."""
    stats = runtime_test_stats
    print(f"\n\nRuntime Testing Coverage Statistics:")
    print(f"  Total problems: {stats['total']}")
    print(f"  Extractable: {stats['extractable']} ({stats['extractable']/stats['total']*100:.1f}%)")
    print(f"  Not extractable: {stats['not_extractable']} ({stats['not_extractable']/stats['total']*100:.1f}%)")
    print(f"  No answer blanks: {stats['no_answers']} ({stats['no_answers']/stats['total']*100:.1f}%)")
    print(f"  Compilation errors: {stats['compilation_errors']} ({stats['compilation_errors']/stats['total']*100:.1f}%)")
    print(f"\n  By category:")
    for category in sorted(stats['by_category'].keys()):
        cat_stats = stats['by_category'][category]
        extractable_pct = (cat_stats['extractable'] / cat_stats['total'] * 100) if cat_stats['total'] > 0 else 0
        print(f"    {category}: {cat_stats['extractable']}/{cat_stats['total']} extractable ({extractable_pct:.1f}%)")
