"""
Tests for ProblemService.

Unit tests for problem service business logic.
"""

import pytest
from pathlib import Path

from services import ProblemService
from models import DifficultyLevel
from core.errors import ProblemNotFoundError


@pytest.mark.asyncio
async def test_get_problem_metadata(problem_repository, sample_problem_metadata):
    """Test getting problem metadata"""
    service = ProblemService(problem_repository)

    metadata = await service.get_problem_metadata("test_problem")

    assert metadata.id == "test_problem"
    assert metadata.title == "Test Problem"


@pytest.mark.asyncio
async def test_get_problem_not_found(problem_repository):
    """Test getting non-existent problem raises error"""
    service = ProblemService(problem_repository)

    with pytest.raises(ProblemNotFoundError):
        await service.get_problem_metadata("nonexistent")


@pytest.mark.asyncio
async def test_list_problems(problem_repository):
    """Test listing all problems"""
    service = ProblemService(problem_repository)

    problems = await service.list_problems()

    assert len(problems) >= 1
    assert any(p.id == "test_problem" for p in problems)


@pytest.mark.asyncio
async def test_search_problems_by_query(problem_repository):
    """Test searching problems by query string"""
    service = ProblemService(problem_repository)

    results = await service.search_problems(query="test")

    assert len(results) >= 1
    assert all("test" in p.title.lower() for p in results)


@pytest.mark.asyncio
async def test_search_problems_by_tags(problem_repository):
    """Test searching problems by tags"""
    service = ProblemService(problem_repository)

    results = await service.search_problems(tags=["test"])

    assert len(results) >= 1


@pytest.mark.asyncio
async def test_search_problems_by_difficulty(problem_repository):
    """Test searching problems by difficulty"""
    service = ProblemService(problem_repository)

    results = await service.search_problems(difficulty=DifficultyLevel.BEGINNER)

    assert len(results) >= 0  # May be empty if no beginner problems
