"""
Pytest configuration and fixtures.

Provides shared fixtures for testing.
"""

import pytest
from pathlib import Path
from typing import AsyncGenerator

from fastapi.testclient import TestClient
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main_v2 import app
from repositories import FileSystemProblemRepository
from models import ProblemMetadata


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client"""
    return TestClient(app)


@pytest.fixture
async def problem_repository(tmp_path: Path) -> FileSystemProblemRepository:
    """Test problem repository with temporary directory"""
    repo = FileSystemProblemRepository(tmp_path)

    # Create a sample problem file
    sample_problem = tmp_path / "test_problem.pg"
    sample_problem.write_text("""
DOCUMENT();
loadMacros("PGstandard.pl", "PGML.pl", "MathObjects.pl");
TEXT(beginproblem());

Context("Numeric");
$a = 2;
$answer = Real($a);

BEGIN_PGML
What is [$a]?

[_]{$answer}
END_PGML

ENDDOCUMENT();
    """)

    return repo


@pytest.fixture
def sample_problem_metadata() -> ProblemMetadata:
    """Sample problem metadata for testing"""
    return ProblemMetadata(
        id="test_problem",
        title="Test Problem",
        description="A simple test problem",
        difficulty="beginner",
        tags=["test", "simple"],
        file_path="/tmp/test_problem.pg"
    )
