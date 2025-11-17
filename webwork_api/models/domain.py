"""
Domain models for WebWork application.

These are the core business entities with validation and behavior.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, validator
from enum import Enum


class DifficultyLevel(str, Enum):
    """Problem difficulty levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ProblemMetadata(BaseModel):
    """Problem metadata and categorization"""
    id: str = Field(..., description="Problem identifier")
    title: str = Field(..., description="Problem title")
    description: Optional[str] = Field(None, description="Problem description")
    difficulty: Optional[DifficultyLevel] = None
    tags: List[str] = Field(default_factory=list, description="Problem tags")
    author: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    file_path: str = Field(..., description="Path to .pg file")

    class Config:
        use_enum_values = True


class AnswerBlank(BaseModel):
    """Information about an answer blank"""
    name: str
    type: str = "text"
    width: int = 20
    correct_answer: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)


class ProblemContent(BaseModel):
    """Rendered problem content"""
    statement_html: str
    answer_blanks: List[AnswerBlank]
    solution_html: Optional[str] = None
    hint_html: Optional[str] = None
    header_html: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    seed: int


class AnswerSubmission(BaseModel):
    """Student answer submission"""
    answer_blank_name: str
    answer_value: str
    submitted_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('answer_value')
    def sanitize_answer(cls, v):
        """Sanitize answer input"""
        if len(v) > 10000:
            raise ValueError("Answer too long (max 10000 characters)")
        return v.strip()


class AnswerFeedback(BaseModel):
    """Feedback for a single answer"""
    answer_blank_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    correct: bool
    student_answer: str
    correct_answer: str
    message: str
    messages: List[str] = Field(default_factory=list)
    preview: Optional[str] = None
    error_message: Optional[str] = None


class SessionState(BaseModel):
    """User session state for a problem"""
    session_id: UUID = Field(default_factory=uuid4)
    user_id: Optional[UUID] = None
    problem_id: str
    seed: int
    answers: Dict[str, str] = Field(default_factory=dict)
    score: Optional[float] = None
    attempts: int = 0
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    state: Dict[str, Any] = Field(default_factory=dict)

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()

    def record_attempt(self, score: float):
        """Record a grading attempt"""
        self.attempts += 1
        self.score = score
        self.submitted_at = datetime.utcnow()
        self.update_activity()


class User(BaseModel):
    """User entity"""
    id: UUID = Field(default_factory=uuid4)
    email: str
    full_name: Optional[str] = None
    role: str = "student"  # student, instructor, admin
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

    @validator('email')
    def validate_email(cls, v):
        """Basic email validation"""
        if '@' not in v or '.' not in v:
            raise ValueError("Invalid email format")
        return v.lower()


class ProblemAttempt(BaseModel):
    """A single attempt at solving a problem"""
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    answers: Dict[str, str]
    results: Dict[str, AnswerFeedback]
    score: float = Field(..., ge=0.0, le=1.0)
    attempted_at: datetime = Field(default_factory=datetime.utcnow)
