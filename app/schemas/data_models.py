"""Data layer models — mirror schemas/*.json and docs/data_schema.md."""

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RoleCode(str, Enum):
    MLE = "MLE"
    AI = "AI"
    DS = "DS"
    SWE = "SWE"
    GENAI = "GENAI"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Topic(str, Enum):
    ML = "ml"
    ML_THEORY = "ml_theory"
    DEEP_LEARNING = "deep_learning"
    MLOPS = "mlops"
    SWE = "swe"
    DSA = "dsa"
    OOP = "oop"
    SYSTEM_DESIGN = "system_design"
    AI = "ai"
    LLMS = "llms"
    RAG = "rag"
    AGENTS = "agents"


class TechnicalTopic(str, Enum):
    PYTHON = "python"
    SQL = "sql"
    PYTORCH = "pytorch"
    TRANSFORMERS = "transformers"
    RAG = "rag"
    LANGGRAPH = "langgraph"
    SYSTEM_DESIGN = "system_design"
    MLOPS = "mlops"
    VECTOR_DATABASES = "vector_databases"


class QuestionType(str, Enum):
    THEORY = "theory"
    CODING = "coding"
    SYSTEM_DESIGN = "system_design"
    BEHAVIORAL = "behavioral"
    ML = "ml"
    SWE = "swe"


class Seniority(str, Enum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"


class JobDescription(BaseModel):
    id: str
    role: RoleCode
    role_title: str
    skills: list[str] = Field(min_length=1)
    requirements: str = Field(min_length=50)
    seniority: Optional[Seniority] = None
    company: Optional[str] = None
    source: str
    source_url: Optional[str] = None
    collected_at: date


class InterviewQuestion(BaseModel):
    id: str
    question: str = Field(min_length=10)
    topic: Topic
    difficulty: Difficulty
    role_tags: list[RoleCode] = Field(min_length=1)
    question_type: QuestionType
    expected_points: list[str] = Field(min_length=1)
    source: str


class TechnicalDocManifest(BaseModel):
    filename: str
    topic: TechnicalTopic
    title: str
    source: str
    license: str
    source_url: Optional[str] = None
