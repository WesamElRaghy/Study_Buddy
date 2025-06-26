from pydantic import BaseModel, Field, EmailStr
from typing import List, Dict, Optional, Literal
from datetime import datetime

# Query-related schemas
class TextbookQuery(BaseModel):
    query: str
    temperature: float = Field(0.3, ge=0.0, le=1.0)

class PracticeQuestionsRequest(BaseModel):
    question: str
    num_questions: int = Field(3, ge=1)
    temperature: float = Field(0.3, ge=0.0, le=1.0)

# Optional: use a typed model for context items
class ContextItem(BaseModel):
    title: str
    content: str

class StructuredQueryResponse(BaseModel):
    question_number: Optional[int]
    question_text: Optional[str]
    context: List[ContextItem]
    task: str
    answer: str
    time_taken: float

# User-related schemas
class UserCreate(BaseModel):
    email: EmailStr
    password: Optional[str] = None  # Optional for Google auth
    full_name: Optional[str] = None
    role: Literal["student", "teacher", "admin"] = "student"
    google_id: Optional[str] = None

class UserResponse(BaseModel):
    uid: str
    email: EmailStr
    full_name: Optional[str]
    role: Literal["student", "teacher", "admin"]
    google_id: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime]

class Token(BaseModel):
    access_token: str
    token_type: str

# Lesson-related schemas
class LessonCreate(BaseModel):
    title: str
    description: Optional[str] = None

class LessonResponse(BaseModel):
    id: int
    title: str
    description: Optional[str]
    teacher_id: int
    created_at: datetime

# Enrollment
class EnrollmentCreate(BaseModel):
    lesson_id: int
