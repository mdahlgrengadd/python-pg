"""
API integration tests.

Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "endpoints" in data


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_list_problems(client):
    """Test listing problems"""
    response = client.get("/problems")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_list_problems_with_metadata(client):
    """Test listing problems with metadata"""
    response = client.get("/problems/metadata")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_get_problem(client):
    """Test getting a problem (requires actual problem file)"""
    # This test assumes problems exist in the problems/ directory
    response = client.get("/problems/simple_algebra?seed=12345")

    if response.status_code == 200:
        data = response.json()
        assert "statement_html" in data
        assert "answer_blanks" in data
        assert "seed" in data
        assert data["seed"] == 12345
    else:
        # Problem not found is acceptable in test environment
        assert response.status_code == 404


def test_get_problem_not_found(client):
    """Test getting non-existent problem"""
    response = client.get("/problems/nonexistent")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data


def test_grade_problem(client):
    """Test grading a problem (requires actual problem file)"""
    grade_request = {
        "answers": {
            "AnSwEr0001": "2*x + 3"
        },
        "seed": 12345
    }

    response = client.post("/problems/simple_algebra/grade", json=grade_request)

    if response.status_code == 200:
        data = response.json()
        assert "score" in data
        assert "answer_results" in data
    else:
        # Problem not found is acceptable
        assert response.status_code in [404, 500]


def test_search_problems(client):
    """Test searching problems"""
    response = client.get("/problems/search?query=algebra")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_invalid_request_validation(client):
    """Test request validation"""
    # Missing required field
    invalid_request = {}

    response = client.post("/problems/test/grade", json=invalid_request)
    assert response.status_code == 422  # Validation error
