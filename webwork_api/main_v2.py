"""
FastAPI backend for WebWork Python - Refactored with Layered Architecture.

This version implements:
- Service layer for business logic
- Repository pattern for data access
- Structured logging
- Comprehensive error handling
- Dependency injection
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from core import (
    settings,
    setup_logging,
    get_logger,
    register_error_handlers,
)
from models import (
    ProblemMetadata,
    ProblemContent,
    AnswerFeedback,
    DifficultyLevel,
)
from repositories import get_problem_repository, ProblemRepositoryInterface
from services import ProblemService, GradingService

# Setup logging
setup_logging()
logger = get_logger(__name__)


# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info(
        "Starting WebWork API",
        extra_data={
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG
        }
    )
    yield
    logger.info("Shutting down WebWork API")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="REST API for WebWork problem rendering and grading with layered architecture",
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan
)

# Register error handlers
register_error_handlers(app)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)


# Dependency injection
def get_problem_service_dep() -> ProblemService:
    """Get problem service instance"""
    repository = get_problem_repository()
    return ProblemService(repository)


def get_grading_service_dep() -> GradingService:
    """Get grading service instance"""
    repository = get_problem_repository()
    return GradingService(repository)


# API Request/Response Models
class GradeRequest(BaseModel):
    """Request to grade a problem"""
    answers: Dict[str, str] = Field(..., description="Student answers by answer blank name")
    seed: Optional[int] = Field(None, description="Random seed (use same seed as when problem was loaded)")


class AnswerFeedbackResponse(BaseModel):
    """Answer grading feedback response"""
    answer_blank_name: str
    score: float
    correct: bool
    student_answer: str
    correct_answer: str
    message: str
    messages: List[str]
    preview: Optional[str]
    error_message: Optional[str]

    @classmethod
    def from_domain(cls, feedback: AnswerFeedback) -> "AnswerFeedbackResponse":
        """Convert domain model to response"""
        return cls(
            answer_blank_name=feedback.answer_blank_name,
            score=feedback.score,
            correct=feedback.correct,
            student_answer=feedback.student_answer,
            correct_answer=feedback.correct_answer,
            message=feedback.message,
            messages=feedback.messages,
            preview=feedback.preview,
            error_message=feedback.error_message,
        )


class GradeResponse(BaseModel):
    """Grading response"""
    score: float
    answer_results: Dict[str, AnswerFeedbackResponse]
    problem_result: Optional[Dict[str, Any]] = None


class ProblemMetadataResponse(BaseModel):
    """Problem metadata response"""
    id: str
    title: str
    description: Optional[str]
    difficulty: Optional[str]
    tags: List[str]
    author: Optional[str]

    @classmethod
    def from_domain(cls, metadata: ProblemMetadata) -> "ProblemMetadataResponse":
        """Convert domain model to response"""
        return cls(
            id=metadata.id,
            title=metadata.title,
            description=metadata.description,
            difficulty=metadata.difficulty.value if metadata.difficulty else None,
            tags=metadata.tags,
            author=metadata.author,
        )


# API Routes

@app.get("/")
async def read_root():
    """API root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "endpoints": {
            "problems": "/problems",
            "problem": "/problems/{problem_id}",
            "grade": "/problems/{problem_id}/grade",
            "search": "/problems/search",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }


@app.get("/problems", response_model=List[str])
async def list_problems(
    service: ProblemService = Depends(get_problem_service_dep)
):
    """List available problems (IDs only for backward compatibility)"""
    logger.info("Listing problems")

    problems = await service.list_problems()
    return [p.id for p in problems]


@app.get("/problems/metadata", response_model=List[ProblemMetadataResponse])
async def list_problems_with_metadata(
    service: ProblemService = Depends(get_problem_service_dep)
):
    """List problems with full metadata"""
    logger.info("Listing problems with metadata")

    problems = await service.list_problems()
    return [ProblemMetadataResponse.from_domain(p) for p in problems]


@app.get("/problems/search", response_model=List[ProblemMetadataResponse])
async def search_problems(
    query: Optional[str] = None,
    tags: Optional[str] = None,  # Comma-separated
    difficulty: Optional[DifficultyLevel] = None,
    service: ProblemService = Depends(get_problem_service_dep)
):
    """Search problems by criteria"""
    logger.info(
        "Searching problems",
        extra_data={
            "query": query,
            "tags": tags,
            "difficulty": difficulty
        }
    )

    tag_list = tags.split(",") if tags else None
    problems = await service.search_problems(query, tag_list, difficulty)

    return [ProblemMetadataResponse.from_domain(p) for p in problems]


@app.get("/problems/{problem_id}", response_model=ProblemContent)
async def get_problem(
    problem_id: str,
    seed: int = 12345,
    service: ProblemService = Depends(get_problem_service_dep)
):
    """
    Get a problem by ID.

    Args:
        problem_id: Problem identifier (filename without .pg)
        seed: Random seed for problem generation

    Returns:
        Rendered problem with answer blanks
    """
    logger.info(
        "Getting problem",
        extra_data={"problem_id": problem_id, "seed": seed}
    )

    return await service.get_problem(problem_id, seed)


@app.post("/problems/{problem_id}/grade", response_model=GradeResponse)
async def grade_problem(
    problem_id: str,
    request: GradeRequest,
    grading_service: GradingService = Depends(get_grading_service_dep)
):
    """
    Grade student answers for a problem.

    Args:
        problem_id: Problem identifier
        request: Grading request with student answers

    Returns:
        Grading results with scores and feedback
    """
    seed = request.seed or 12345

    logger.info(
        "Grading problem",
        extra_data={
            "problem_id": problem_id,
            "seed": seed,
            "num_answers": len(request.answers)
        }
    )

    score, feedback_dict, problem_result = await grading_service.grade_answers(
        problem_id,
        request.answers,
        seed
    )

    # Convert feedback to response models
    answer_results = {
        name: AnswerFeedbackResponse.from_domain(feedback)
        for name, feedback in feedback_dict.items()
    }

    return GradeResponse(
        score=score,
        answer_results=answer_results,
        problem_result=problem_result
    )


@app.get("/problems/{problem_id}/metadata", response_model=ProblemMetadataResponse)
async def get_problem_metadata(
    problem_id: str,
    service: ProblemService = Depends(get_problem_service_dep)
):
    """Get problem metadata only (without rendering)"""
    logger.info(
        "Getting problem metadata",
        extra_data={"problem_id": problem_id}
    )

    metadata = await service.get_problem_metadata(problem_id)
    return ProblemMetadataResponse.from_domain(metadata)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main_v2:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
