"""
Comprehensive test suite for all tutorial sample problems.

Tests all .pg files in tutorial/sample-problems directory to ensure they render
without compilation errors using the same method as pg_solve.py and pg-convert.

Usage:
    # Run all problems
    pytest pg/translator/tests/test_tutorial_sample_problems_all.py -v

    # Run problems by category (case-insensitive)
    pytest pg/translator/tests/test_tutorial_sample_problems_all.py -v -k "Arithmetic"
    pytest pg/translator/tests/test_tutorial_sample_problems_all.py -v -k "Algebra"

    # Run specific problem by name (case-insensitive)
    pytest pg/translator/tests/test_tutorial_sample_problems_all.py -v -k "UnitConversion"
    pytest pg/translator/tests/test_tutorial_sample_problems_all.py -v -k "ExpandedPolynomial"
    pytest pg/translator/tests/test_tutorial_sample_problems_all.py -v -k "DifferentiateFunction"

    # Run with detailed output
    pytest pg/translator/tests/test_tutorial_sample_problems_all.py::TestTutorialSampleProblems::test_problem_renders -v --tb=short -k "UnitConversion"
    pytest pg/translator/tests/test_tutorial_sample_problems_all.py::TestTutorialSampleProblems::test_problem_renders -v --tb=long -k "GraphsInTables"

    # Run aggregate test (all problems with summary)
    pytest pg/translator/tests/test_tutorial_sample_problems_all.py::TestTutorialSampleProblems::test_all_sample_problems_render -v

    # Run specific test only
    pytest pg/translator/tests/test_tutorial_sample_problems_all.py::TestTutorialSampleProblems::test_sample_problems_contain_files -v

Examples:
    # Test the UnitConversion problem
    pytest pg/translator/tests/test_tutorial_sample_problems_all.py -v -k "UnitConversion"

    # Test all Arithmetic problems
    pytest pg/translator/tests/test_tutorial_sample_problems_all.py -v -k "Arithmetic"

    # Test all DiffCalc problems
    pytest pg/translator/tests/test_tutorial_sample_problems_all.py -v -k "DiffCalc"

    # Test failing problems
    pytest pg/translator/tests/test_tutorial_sample_problems_all.py -v -k "AnswerWithUnits or DifferentiateFunction or GraphsInTables"

    # See what problems are available (without running tests)
    pytest pg/translator/tests/test_tutorial_sample_problems_all.py --collect-only -q
"""

import pytest
from pathlib import Path
import sys
from typing import List, Tuple

# Add packages to path
_pg_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_pg_root / "packages"))

from pg.translator import PGTranslator


class TestTutorialSampleProblems:
    """Test suite for all tutorial sample problems."""

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
        # Example: tutorial/sample-problems/Arithmetic/UnitConversion.pg
        # Returns: ("Arithmetic", "UnitConversion")
        parts = problem_path.relative_to(problem_path.parent.parent.parent.parent).parts
        if len(parts) >= 3:
            category = parts[2]
            name = problem_path.stem
            return category, name
        return problem_path.parent.name, problem_path.stem

    def test_all_sample_problems_render(self):
        """
        Test that all sample problems render without compilation errors.

        This is the main test that runs against all discovered problem files.
        """
        problem_files = self.get_all_problem_files()

        if not problem_files:
            pytest.skip("No problem files found")

        translator = PGTranslator()
        results = []
        failures = []

        for problem_path in problem_files:
            category, name = self.get_category_and_name(problem_path)
            test_id = f"{category}/{name}"

            try:
                # Translate the problem with a fixed seed for reproducibility
                result = translator.translate(str(problem_path), seed=1234)

                # Check for errors
                if result.errors:
                    failure_msg = f"{test_id}: {result.errors[0]}"
                    failures.append(failure_msg)
                    results.append((test_id, "ERROR", result.errors[0]))
                else:
                    results.append((test_id, "PASS", None))

            except Exception as e:
                failure_msg = f"{test_id}: {type(e).__name__}: {str(e)[:100]}"
                failures.append(failure_msg)
                results.append((test_id, "FAIL", str(e)))

        # Print summary
        print(f"\n\n{'='*80}")
        print(f"TUTORIAL SAMPLE PROBLEMS TEST SUMMARY")
        print(f"{'='*80}\n")

        passed = len([r for r in results if r[1] == "PASS"])
        failed = len([r for r in results if r[1] in ("ERROR", "FAIL")])
        total = len(results)

        for test_id, status, error in sorted(results):
            if status == "PASS":
                print(f"  ✓ {test_id}")
            else:
                print(f"  ✗ {test_id}: {status}")
                if error:
                    error_line = error.split('\n')[0][:70]
                    print(f"      {error_line}")

        print(f"\n{'='*80}")
        print(f"Results: {passed} passed, {failed} failed out of {total} total")
        print(f"Success rate: {passed/total*100:.1f}%")
        print(f"{'='*80}\n")

        # Fail the test if any problems have errors
        assert len(failures) == 0, f"{len(failures)} problems failed to render:\n" + "\n".join(failures)

    @pytest.mark.parametrize(
        "problem_path",
        get_all_problem_files(),
        ids=lambda p: str(p.relative_to(_pg_root / "tutorial" / "sample-problems")).replace("\\", "/")
    )
    def test_problem_renders(self, problem_path):
        """
        Parametrized test for each individual problem file.

        This allows pytest to run each problem as a separate test case, with
        individual pass/fail reporting for each problem.
        """
        translator = PGTranslator()

        # Translate the problem
        result = translator.translate(str(problem_path), seed=1234)

        # Check that translation succeeded
        if result.errors:
            error_msg = "\n".join(result.errors[:3])
            pytest.fail(f"Problem rendering failed with errors:\n{error_msg}")

        # Basic checks
        assert result.statement_html is not None, "Problem should have statement_html"
        assert isinstance(result.statement_html, str), "statement_html should be a string"

        # Allow empty statement for snippet problems (they demonstrate techniques, not full problems)
        problem_content = problem_path.read_text()
        is_snippet = "type = snippet" in problem_content
        if not is_snippet:
            assert len(result.statement_html) > 0, "statement_html should not be empty"

    def test_sample_problems_directory_exists(self):
        """Verify that the tutorial/sample-problems directory exists."""
        tutorial_dir = _pg_root / "tutorial" / "sample-problems"
        assert tutorial_dir.exists(), f"tutorial/sample-problems directory not found at {tutorial_dir}"
        assert tutorial_dir.is_dir(), f"tutorial/sample-problems is not a directory"

    def test_sample_problems_contain_files(self):
        """Verify that tutorial/sample-problems contains at least some .pg files."""
        problem_files = self.get_all_problem_files()
        assert len(problem_files) > 0, "No .pg files found in tutorial/sample-problems"
        print(f"\nFound {len(problem_files)} .pg files in tutorial/sample-problems")


# Fixtures for advanced testing scenarios

@pytest.fixture
def translator():
    """Provide a PGTranslator instance for tests."""
    return PGTranslator()


@pytest.fixture
def problem_files():
    """Provide list of all problem files."""
    return TestTutorialSampleProblems.get_all_problem_files()


@pytest.fixture
def sample_problems_stats(problem_files):
    """Generate statistics about the sample problems."""
    categories = {}
    for problem_path in problem_files:
        category, _ = TestTutorialSampleProblems.get_category_and_name(problem_path)
        if category not in categories:
            categories[category] = 0
        categories[category] += 1
    return {
        "total": len(problem_files),
        "categories": categories,
        "most_common": max(categories.items(), key=lambda x: x[1])[0] if categories else None
    }


def test_sample_problems_stats(sample_problems_stats):
    """Report statistics about sample problems."""
    stats = sample_problems_stats
    print(f"\n\nSample Problems Statistics:")
    print(f"  Total problems: {stats['total']}")
    print(f"  Categories: {len(stats['categories'])}")
    for category, count in sorted(stats['categories'].items()):
        print(f"    - {category}: {count} problems")
    if stats['most_common']:
        print(f"  Most common category: {stats['most_common']}")
