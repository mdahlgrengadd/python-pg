# WebWork Python - Architecture & Design Principles

This document outlines the architecture, design patterns, and principles for a robust, production-ready WebWork Python system.

## 🎯 Core Principles

### 1. Separation of Concerns
- **Backend**: Clean layered architecture (API → Service → Repository → Domain)
- **Frontend**: Component-based architecture with clear responsibilities
- **Data Flow**: Unidirectional data flow with clear state management

### 2. Type Safety
- **Backend**: Pydantic models for all data structures
- **Frontend**: TypeScript with strict mode enabled
- **API**: OpenAPI schema validation

### 3. Error Handling
- **Graceful Degradation**: System continues to function with partial failures
- **User-Friendly Messages**: Clear, actionable error messages
- **Logging**: All errors logged with context for debugging

### 4. Testability
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **E2E Tests**: Test complete user workflows

### 5. Performance
- **Caching**: Intelligent caching at multiple layers
- **Lazy Loading**: Load resources only when needed
- **Code Splitting**: Split frontend bundles for faster loading

### 6. Security
- **Input Validation**: All inputs validated and sanitized
- **Rate Limiting**: Protect against abuse
- **CORS**: Properly configured for production
- **Content Security Policy**: Prevent XSS attacks

---

## 🏗️ Backend Architecture

### Layered Architecture

```
┌─────────────────────────────────────────────────┐
│                  API Layer                      │
│  (FastAPI Routes, Request/Response Handling)    │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│               Service Layer                     │
│  (Business Logic, Problem Translation, Grading) │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│             Repository Layer                    │
│  (Data Access, Problem Storage, Caching)        │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┘
│               Domain Layer                      │
│  (Pydantic Models, Business Entities)           │
└─────────────────────────────────────────────────┘
```

### Key Components

#### 1. API Layer (`webwork_api/api/`)
- **Routes**: HTTP endpoint definitions
- **Dependencies**: FastAPI dependency injection
- **Middleware**: CORS, logging, error handling
- **Validators**: Request/response validation

#### 2. Service Layer (`webwork_api/services/`)
- **ProblemService**: Problem loading and rendering
- **GradingService**: Answer validation and scoring
- **CacheService**: Intelligent caching strategy
- **SessionService**: User session management

#### 3. Repository Layer (`webwork_api/repositories/`)
- **ProblemRepository**: Problem CRUD operations
- **UserRepository**: User data access
- **SessionRepository**: Session persistence

#### 4. Domain Layer (`webwork_api/models/`)
- **Problem**: Problem entity with metadata
- **Answer**: Answer model with validation
- **User**: User entity
- **Session**: Session state

### Design Patterns

#### Repository Pattern
```python
class ProblemRepository:
    """Abstract data access for problems"""

    async def get(self, problem_id: str) -> Problem:
        """Get problem by ID"""

    async def list(self) -> List[Problem]:
        """List all problems"""

    async def search(self, query: str) -> List[Problem]:
        """Search problems"""
```

#### Service Pattern
```python
class ProblemService:
    """Business logic for problems"""

    def __init__(self, repo: ProblemRepository, cache: CacheService):
        self.repo = repo
        self.cache = cache

    async def get_problem(self, id: str, seed: int) -> ProblemResponse:
        """Get problem with caching and error handling"""
```

#### Dependency Injection
```python
# FastAPI dependency injection
def get_problem_service() -> ProblemService:
    repo = get_problem_repository()
    cache = get_cache_service()
    return ProblemService(repo, cache)

@app.get("/problems/{id}")
async def get_problem(
    id: str,
    service: ProblemService = Depends(get_problem_service)
):
    return await service.get_problem(id)
```

---

## 🎨 Frontend Architecture

### Component Hierarchy

```
App
├── Providers (Context, Theme, Auth)
│   └── Router
│       ├── ProblemListPage
│       │   └── ProblemCard[]
│       ├── ProblemPage
│       │   ├── ProblemHeader
│       │   ├── ProblemStatement
│       │   │   └── MathRenderer
│       │   ├── AnswerSection
│       │   │   ├── AnswerInput[]
│       │   │   └── SubmitButton
│       │   ├── FeedbackSection
│       │   │   └── AnswerFeedback[]
│       │   ├── HintSection (collapsible)
│       │   └── SolutionSection (collapsible)
│       └── AdminPage
│           ├── ProblemManager
│           └── Analytics
```

### State Management

#### Context API Structure
```typescript
// Global state contexts
AppContext          // App-level state
├── AuthContext     // User authentication
├── ThemeContext    // UI theme
├── ProblemContext  // Current problem state
└── CacheContext    // Client-side caching
```

#### Custom Hooks
```typescript
// Domain-specific hooks
useProblem(id, seed)     // Fetch and manage problem
useGrading()              // Submit and manage answers
useAuth()                 // Authentication state
useCache()                // Client-side caching
usePersistence()          // LocalStorage persistence
```

### Design Patterns

#### Container/Presentational Pattern
```typescript
// Container: Logic and state management
const ProblemContainer: FC = () => {
  const { problem, loading, error } = useProblem(id, seed);
  const { submit, grading } = useGrading();

  return <ProblemView
    problem={problem}
    onSubmit={submit}
    loading={loading || grading}
  />;
};

// Presentational: Pure UI component
const ProblemView: FC<Props> = ({ problem, onSubmit, loading }) => {
  return <div>{/* Pure presentation */}</div>;
};
```

#### Compound Components Pattern
```typescript
// Flexible, composable components
<Problem>
  <Problem.Header />
  <Problem.Statement />
  <Problem.Answers>
    <Problem.AnswerInput name="ans1" />
    <Problem.AnswerInput name="ans2" />
  </Problem.Answers>
  <Problem.Submit />
  <Problem.Feedback />
</Problem>
```

---

## 🔒 Security Architecture

### Input Validation

#### Backend
```python
from pydantic import BaseModel, validator, Field

class AnswerInput(BaseModel):
    answer: str = Field(..., max_length=1000)

    @validator('answer')
    def sanitize_answer(cls, v):
        # Remove potentially dangerous characters
        # Validate against injection attacks
        return sanitize(v)
```

#### Frontend
```typescript
// Input sanitization
const sanitizeInput = (input: string): string => {
  return DOMPurify.sanitize(input);
};
```

### Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/problems/{id}")
@limiter.limit("100/minute")
async def get_problem(id: str):
    ...
```

### CORS Configuration
```python
# Production CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://webwork.example.com",
        "https://www.example.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## 📊 Data Architecture

### Database Schema

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Problems table (metadata only, .pg files stored in filesystem)
CREATE TABLE problems (
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    difficulty VARCHAR(50),
    tags JSONB,
    file_path VARCHAR(1000) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Sessions table
CREATE TABLE sessions (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    problem_id VARCHAR(255) REFERENCES problems(id),
    seed INTEGER NOT NULL,
    answers JSONB,
    score FLOAT,
    started_at TIMESTAMP DEFAULT NOW(),
    submitted_at TIMESTAMP,
    state JSONB
);

-- Attempts table
CREATE TABLE attempts (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES sessions(id),
    answers JSONB NOT NULL,
    results JSONB NOT NULL,
    score FLOAT NOT NULL,
    attempted_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_problem ON sessions(problem_id);
CREATE INDEX idx_attempts_session ON attempts(session_id);
```

### Caching Strategy

```
┌─────────────────────────────────────────────────┐
│              Caching Layers                     │
├─────────────────────────────────────────────────┤
│ 1. Browser Cache (Service Worker)              │
│    - Static assets (JS, CSS, images)           │
│    - Problem HTML (with TTL)                    │
├─────────────────────────────────────────────────┤
│ 2. Application Cache (React State)             │
│    - Current problem state                      │
│    - User answers                               │
├─────────────────────────────────────────────────┤
│ 3. API Cache (Redis)                            │
│    - Rendered problems (by id:seed)             │
│    - User sessions                              │
│    - Problem lists                              │
├─────────────────────────────────────────────────┤
│ 4. Database Query Cache (PostgreSQL)           │
│    - Frequently accessed data                   │
└─────────────────────────────────────────────────┘
```

---

## 🧪 Testing Strategy

### Test Pyramid

```
        ┌──────────────┐
        │     E2E      │  ← 10% (Full user workflows)
        │   (Cypress)  │
       ┌┴──────────────┴┐
       │   Integration   │  ← 30% (Component interactions)
       │  (Pytest, RTL)  │
      ┌┴─────────────────┴┐
      │    Unit Tests      │  ← 60% (Individual functions)
      │ (Pytest, Jest)     │
      └────────────────────┘
```

### Backend Tests
```python
# Unit test example
def test_problem_service_get():
    repo = MockProblemRepository()
    service = ProblemService(repo)

    problem = await service.get_problem("test", 12345)

    assert problem.statement_html is not None
    assert len(problem.answer_blanks) > 0

# Integration test
async def test_problem_endpoint():
    async with AsyncClient(app=app) as client:
        response = await client.get("/problems/simple_algebra")
        assert response.status_code == 200
        data = response.json()
        assert "statement_html" in data
```

### Frontend Tests
```typescript
// Unit test
test('AnswerInput handles change', () => {
  const onChange = jest.fn();
  render(<AnswerInput name="ans1" onChange={onChange} />);

  fireEvent.change(screen.getByRole('textbox'), {
    target: { value: '2*x + 3' }
  });

  expect(onChange).toHaveBeenCalledWith('ans1', '2*x + 3');
});

// Integration test
test('Problem submission workflow', async () => {
  render(<Problem id="test" seed={12345} />);

  const input = screen.getByRole('textbox');
  fireEvent.change(input, { target: { value: '42' } });

  fireEvent.click(screen.getByText('Submit'));

  await waitFor(() => {
    expect(screen.getByText('Correct!')).toBeInTheDocument();
  });
});
```

---

## 📈 Monitoring & Observability

### Logging Structure
```python
import structlog

logger = structlog.get_logger()

logger.info(
    "problem_loaded",
    problem_id=problem_id,
    seed=seed,
    user_id=user_id,
    duration_ms=duration
)
```

### Metrics to Track
- **Performance**: API response times, render times
- **Usage**: Problems viewed, answers submitted, success rates
- **Errors**: Error rates, error types
- **Business**: User engagement, completion rates

### Health Checks
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": await check_database(),
        "cache": await check_cache(),
        "translator": await check_translator()
    }
```

---

## 🚀 Deployment Architecture

### Containerization
```
┌─────────────────────────────────────────┐
│         Load Balancer (Nginx)           │
└────────────┬────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼────┐       ┌────▼───┐
│ API    │       │ API    │
│ Server │       │ Server │
│ (x3)   │       │ (x3)   │
└───┬────┘       └────┬───┘
    │                 │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │                 │
┌───▼─────┐    ┌──────▼──┐
│PostgreSQL│    │  Redis  │
│(Primary) │    │ (Cache) │
└──────────┘    └─────────┘
```

### Docker Compose
```yaml
version: '3.8'

services:
  api:
    build: ./webwork_api
    environment:
      - DATABASE_URL=postgresql://...
      - REDIS_URL=redis://cache:6379
    depends_on:
      - db
      - cache

  frontend:
    build: ./webwork-frontend
    depends_on:
      - api

  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data

  cache:
    image: redis:7

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - api
      - frontend
```

---

## 🔄 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI/CD

on: [push, pull_request]

jobs:
  test-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --cov
      - name: Lint
        run: ruff check .

  test-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Node.js
        uses: actions/setup-node@v3
      - name: Install dependencies
        run: npm ci
      - name: Run tests
        run: npm test
      - name: Lint
        run: npm run lint

  deploy:
    needs: [test-backend, test-frontend]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: ./deploy.sh
```

---

## 📝 Code Quality Standards

### Python (Backend)
- **Linter**: Ruff
- **Formatter**: Black
- **Type Checker**: mypy
- **Test Coverage**: >80%

### TypeScript (Frontend)
- **Linter**: ESLint
- **Formatter**: Prettier
- **Type Checker**: tsc --noEmit
- **Test Coverage**: >70%

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
  - repo: https://github.com/psf/black
    hooks:
      - id: black
  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy
```

---

## 🎯 Next Steps for Implementation

1. **Phase 1: Foundation**
   - [ ] Implement layered architecture
   - [ ] Add comprehensive error handling
   - [ ] Set up logging infrastructure
   - [ ] Create database schema

2. **Phase 2: Robustness**
   - [ ] Add unit tests (>80% coverage)
   - [ ] Implement caching layer
   - [ ] Add input validation and sanitization
   - [ ] Set up monitoring

3. **Phase 3: Scale**
   - [ ] Docker containerization
   - [ ] CI/CD pipeline
   - [ ] Load testing
   - [ ] Performance optimization

4. **Phase 4: Production**
   - [ ] Security audit
   - [ ] Documentation
   - [ ] Deployment automation
   - [ ] Monitoring dashboards
