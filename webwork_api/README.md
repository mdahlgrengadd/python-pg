# WebWork Python API

FastAPI backend for WebWork Python problems with Pydantic models and REST endpoints.

## Features

- **FastAPI** framework for high performance
- **Pydantic v2** models for data validation
- **REST API** with automatic OpenAPI documentation
- **CORS support** for frontend integration
- **Problem rendering** with PG file translation
- **Answer grading** with detailed feedback
- **Type safety** throughout the stack

## Prerequisites

- Python 3.10+
- WebWork Python package installed (parent directory)

## Installation

```bash
cd webwork_api
pip install -r requirements.txt
```

Make sure the parent `pg` package is importable:

```bash
# From the python-pg root directory
pip install -e .
```

## Running the API

### Development Mode

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: `http://localhost:8000`
- **Docs**: `http://localhost:8000/docs` (Swagger UI)
- **ReDoc**: `http://localhost:8000/redoc` (Alternative docs)

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### `GET /`
Root endpoint with API information.

**Response:**
```json
{
  "name": "WebWork Python API",
  "version": "1.0.0",
  "endpoints": {
    "problems": "/problems/{problem_id}",
    "grade": "/problems/{problem_id}/grade",
    "list": "/problems"
  }
}
```

### `GET /problems`
List all available problems.

**Response:**
```json
["simple_algebra", "quadratic", "calculus_derivative"]
```

### `GET /problems/{problem_id}?seed={seed}`
Get a rendered problem.

**Parameters:**
- `problem_id` (path): Problem file name without `.pg`
- `seed` (query, optional): Random seed (default: 12345)

**Response:**
```json
{
  "statement_html": "<p>Solve: ...</p>",
  "answer_blanks": [
    {
      "name": "AnSwEr0001",
      "type": "formula",
      "width": 20,
      "correct_answer": "2*x + 3"
    }
  ],
  "solution_html": "<p>Solution...</p>",
  "hint_html": "<p>Hint...</p>",
  "header_html": "",
  "metadata": {
    "seed": 12345,
    "num_answers": 1,
    "display_mode": "HTML"
  },
  "errors": null,
  "warnings": null,
  "seed": 12345
}
```

### `POST /problems/{problem_id}/grade`
Grade student answers.

**Parameters:**
- `problem_id` (path): Problem identifier

**Request Body:**
```json
{
  "answers": {
    "AnSwEr0001": "2*x + 3"
  },
  "seed": 12345
}
```

**Response:**
```json
{
  "score": 1.0,
  "answer_results": {
    "AnSwEr0001": {
      "score": 1.0,
      "correct": true,
      "student_answer": "2*x+3",
      "student_correct_answer": "2*x+3",
      "answer_message": "Correct!",
      "messages": [],
      "type": "Value (Formula)",
      "preview": "2x+3",
      "error_message": "",
      "error_flag": false,
      "ans_label": "AnSwEr0001"
    }
  },
  "problem_result": {
    "score": 1.0,
    "type": "std_problem_grader",
    "msg": ""
  },
  "problem_state": {
    "recorded_score": 1.0,
    "num_of_correct_ans": 1,
    "num_of_incorrect_ans": 0
  }
}
```

### `DELETE /cache`
Clear the problem cache (useful during development).

**Response:**
```json
{
  "message": "Cache cleared"
}
```

## Data Models

### ProblemResponse (Pydantic)
```python
class ProblemResponse(BaseModel):
    statement_html: str
    answer_blanks: List[AnswerBlankInfo]
    solution_html: Optional[str]
    hint_html: Optional[str]
    header_html: Optional[str]
    metadata: Optional[Dict[str, Any]]
    errors: Optional[List[str]]
    warnings: Optional[List[str]]
    seed: int
```

### AnswerResultResponse (Pydantic)
```python
class AnswerResultResponse(BaseModel):
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
```

### GradeRequest (Pydantic)
```python
class GradeRequest(BaseModel):
    answers: Dict[str, str]
    seed: Optional[int] = None
```

### GradeResponse (Pydantic)
```python
class GradeResponse(BaseModel):
    score: float
    answer_results: Dict[str, AnswerResultResponse]
    problem_result: Optional[Dict[str, Any]]
    problem_state: Optional[Dict[str, Any]]
```

## Adding Problems

Place `.pg` files in the `problems/` directory:

```
webwork_api/
└── problems/
    ├── simple_algebra.pg
    ├── quadratic.pg
    └── calculus_derivative.pg
```

Problems should follow standard PG format:

```perl
DOCUMENT();
loadMacros("PGstandard.pl", "PGML.pl", "MathObjects.pl");
TEXT(beginproblem());

Context("Numeric");
$a = random(2, 9, 1);
$answer = Formula("$a * x");

BEGIN_PGML
Simplify: [` [$a]x `]

[_]{$answer}
END_PGML

ENDDOCUMENT();
```

## Architecture

```
┌─────────────┐
│   Client    │
│  (Browser)  │
└──────┬──────┘
       │
       │ HTTP/JSON
       │
┌──────▼──────────┐
│   FastAPI App   │
│   (main.py)     │
├─────────────────┤
│ Pydantic Models │
│ - Problem       │
│ - AnswerResult  │
│ - Grade         │
└──────┬──────────┘
       │
       │ Uses
       │
┌──────▼──────────┐
│  PGTranslator   │
│  (pg.translator)│
├─────────────────┤
│ - Parse .pg     │
│ - Execute code  │
│ - Render HTML   │
│ - Grade answers │
└─────────────────┘
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- **200 OK**: Successful request
- **404 Not Found**: Problem doesn't exist
- **500 Internal Server Error**: Problem rendering/grading error

Error response format:
```json
{
  "detail": "Error message with traceback..."
}
```

## CORS Configuration

By default, CORS is configured to allow all origins for development:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Production**: Update `allow_origins` to specific domains.

## Caching

The API implements a simple in-memory cache for rendered problems:

- Key format: `{problem_id}:{seed}`
- Cleared on server restart
- Can be manually cleared via `DELETE /cache`

**Production**: Use Redis or similar for distributed caching.

## Development Tips

### Auto-reload
FastAPI's `--reload` flag watches for file changes:
```bash
uvicorn main:app --reload
```

### Interactive Docs
Visit `/docs` for interactive API testing with Swagger UI.

### Testing with curl

Get a problem:
```bash
curl http://localhost:8000/problems/simple_algebra?seed=12345
```

Grade answers:
```bash
curl -X POST http://localhost:8000/problems/simple_algebra/grade \
  -H "Content-Type: application/json" \
  -d '{"answers": {"AnSwEr0001": "2*x + 3"}, "seed": 12345}'
```

## Performance Considerations

- **Async**: FastAPI is async-ready (currently using sync functions)
- **Workers**: Use multiple uvicorn workers in production
- **Caching**: Implement Redis for production caching
- **Database**: Add PostgreSQL for problem storage and user sessions

## Security

**Important for production:**

1. **CORS**: Restrict allowed origins
2. **Rate limiting**: Add rate limiting middleware
3. **Authentication**: Add user authentication
4. **Input validation**: Already handled by Pydantic
5. **Sandboxing**: PG executor already uses RestrictedPython

## Monitoring

Add logging and monitoring:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/problems/{problem_id}")
async def get_problem(problem_id: str):
    logger.info(f"Loading problem: {problem_id}")
    # ... rest of code
```

## License

Same as WebWork Python (check main repository)
