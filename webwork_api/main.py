"""
FastAPI backend for WebWork Python.

Provides REST API endpoints for:
- Getting problems
- Grading student answers
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from pathlib import Path
import traceback

from pg.translator.translator import PGTranslator, ProblemResult
from pg.answer.answer_hash import AnswerResult

app = FastAPI(
    title="WebWork Python API",
    description="REST API for WebWork problem rendering and grading",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize translator
translator = PGTranslator()

# Example problems directory
PROBLEMS_DIR = Path(__file__).parent / "problems"
PROBLEMS_DIR.mkdir(exist_ok=True)


# API Models
class AnswerInput(BaseModel):
    """Student answer input"""
    answer: str = Field(..., description="Student's answer")


class GradeRequest(BaseModel):
    """Request to grade a problem"""
    answers: Dict[str, str] = Field(..., description="Student answers by answer blank name")
    seed: Optional[int] = Field(None, description="Random seed (use same seed as when problem was loaded)")


class AnswerResultResponse(BaseModel):
    """Answer grading result"""
    score: float
    correct: bool
    student_answer: str
    student_correct_answer: str
    answer_message: str
    messages: List[str]
    type: str
    preview: str
    error_message: str
    error_flag: bool
    ans_label: str

    @classmethod
    def from_answer_result(cls, result: AnswerResult) -> "AnswerResultResponse":
        """Convert AnswerResult to API response"""
        return cls(
            score=result.score,
            correct=result.correct,
            student_answer=result.student_answer,
            student_correct_answer=result.student_correct_answer,
            answer_message=result.answer_message,
            messages=result.messages,
            type=result.type,
            preview=result.preview,
            error_message=result.error_message,
            error_flag=result.error_flag,
            ans_label=result.ans_label,
        )


class AnswerBlankInfo(BaseModel):
    """Information about an answer blank"""
    name: str
    type: str = "text"
    width: int = 20
    correct_answer: Optional[str] = None


class ProblemResponse(BaseModel):
    """Problem rendering response"""
    statement_html: str
    answer_blanks: List[AnswerBlankInfo]
    solution_html: Optional[str] = None
    hint_html: Optional[str] = None
    header_html: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    seed: int


class GradeResponse(BaseModel):
    """Grading response"""
    score: float
    answer_results: Dict[str, AnswerResultResponse]
    problem_result: Optional[Dict[str, Any]] = None
    problem_state: Optional[Dict[str, Any]] = None


# In-memory problem cache (in production, use Redis or similar)
problem_cache: Dict[str, ProblemResult] = {}


@app.get("/")
def read_root():
    """API root endpoint"""
    return {
        "name": "WebWork Python API",
        "version": "1.0.0",
        "endpoints": {
            "problems": "/problems/{problem_id}",
            "grade": "/problems/{problem_id}/grade",
            "list": "/problems"
        }
    }


@app.get("/problems", response_model=List[str])
def list_problems():
    """List available problems"""
    problems = []
    if PROBLEMS_DIR.exists():
        for pg_file in PROBLEMS_DIR.glob("*.pg"):
            problems.append(pg_file.stem)
    return problems


@app.get("/problems/{problem_id}", response_model=ProblemResponse)
def get_problem(problem_id: str, seed: int = 12345):
    """
    Get a problem by ID.

    Args:
        problem_id: Problem identifier (filename without .pg)
        seed: Random seed for problem generation

    Returns:
        Rendered problem with answer blanks
    """
    # Find problem file
    pg_file = PROBLEMS_DIR / f"{problem_id}.pg"

    if not pg_file.exists():
        raise HTTPException(status_code=404, detail=f"Problem '{problem_id}' not found")

    try:
        # Translate problem
        result = translator.translate(pg_file, seed=seed)

        # Cache the result for grading
        cache_key = f"{problem_id}:{seed}"
        problem_cache[cache_key] = result

        # Extract answer blank information
        answer_blanks = []
        for name, blank_info in result.answer_blanks.items():
            evaluator = blank_info.get("evaluator") if isinstance(blank_info, dict) else blank_info

            # Get answer type
            answer_type = "text"
            if hasattr(evaluator, "__class__"):
                answer_type = evaluator.__class__.__name__.lower()

            # Get correct answer for display (if showing solutions)
            correct_answer = None
            if hasattr(evaluator, "__str__"):
                correct_answer = str(evaluator)

            answer_blanks.append(AnswerBlankInfo(
                name=name,
                type=answer_type,
                width=20,
                correct_answer=correct_answer
            ))

        return ProblemResponse(
            statement_html=result.statement_html,
            answer_blanks=answer_blanks,
            solution_html=result.solution_html,
            hint_html=result.hint_html,
            header_html=result.header_html,
            metadata=result.metadata,
            errors=result.errors,
            warnings=result.warnings,
            seed=seed
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error rendering problem: {str(e)}\n{error_trace}"
        )


@app.post("/problems/{problem_id}/grade", response_model=GradeResponse)
def grade_problem(problem_id: str, request: GradeRequest):
    """
    Grade student answers for a problem.

    Args:
        problem_id: Problem identifier
        request: Grading request with student answers

    Returns:
        Grading results with scores and feedback
    """
    seed = request.seed or 12345
    cache_key = f"{problem_id}:{seed}"

    # Find problem file
    pg_file = PROBLEMS_DIR / f"{problem_id}.pg"

    if not pg_file.exists():
        raise HTTPException(status_code=404, detail=f"Problem '{problem_id}' not found")

    try:
        # Translate problem with student answers
        result = translator.translate(
            pg_file,
            seed=seed,
            inputs=request.answers
        )

        # Convert answer results to response format
        answer_results = {}
        if result.answer_results:
            for name, ans_result in result.answer_results.items():
                answer_results[name] = AnswerResultResponse.from_answer_result(ans_result)

        # Calculate overall score
        score = result.score or 0.0

        return GradeResponse(
            score=score,
            answer_results=answer_results,
            problem_result=result.problem_result,
            problem_state=result.problem_state
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error grading problem: {str(e)}\n{error_trace}"
        )


@app.delete("/cache")
def clear_cache():
    """Clear the problem cache"""
    problem_cache.clear()
    return {"message": "Cache cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
