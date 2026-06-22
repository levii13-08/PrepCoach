"""LangGraph state — see docs/state_schema.md."""

from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.schemas.data_models import Difficulty, RoleCode, Topic
from app.schemas.evaluation import EvaluationScore


class SessionStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABORTED = "aborted"


class RoundType(str, Enum):
    ML = "ml"
    THEORY = "theory"
    CODING = "coding"
    SYSTEM_DESIGN = "system_design"
    DSA = "dsa"
    OOP = "oop"
    BEHAVIORAL = "behavioral"
    SWE = "swe"


class ContextChunk(BaseModel):
    content: str
    collection: str
    doc_id: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoundRecord(BaseModel):
    round: int
    round_type: RoundType
    topic: Optional[str] = None
    difficulty: Difficulty
    question: str
    user_answer: str
    expected_points: list[str]
    scores: EvaluationScore
    retrieved_doc_ids: list[str] = Field(default_factory=list)


class InterviewState(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    role: RoleCode
    target_job_id: Optional[str] = None
    session_status: SessionStatus = SessionStatus.ACTIVE
    current_round: int = 0
    max_rounds: int = 5
    round_type: RoundType = RoundType.THEORY
    difficulty: Difficulty = Difficulty.MEDIUM
    topic: Optional[Topic] = None
    question: Optional[str] = None
    expected_points: list[str] = Field(default_factory=list)
    user_answer: Optional[str] = None
    retrieved_context: list[ContextChunk] = Field(default_factory=list)
    scores: Optional[EvaluationScore] = None
    history: list[RoundRecord] = Field(default_factory=list)
    skill_signals: dict[str, float] = Field(default_factory=dict)
    coach_feedback: Optional[str] = None
    skill_gap_report: Optional[dict[str, Any]] = None
    learning_roadmap: Optional[dict[str, Any]] = None
    flags: dict[str, Any] = Field(default_factory=dict)

    def update_skill_signal(self, topic_key: str, score: float, alpha: float = 0.3) -> None:
        prev = self.skill_signals.get(topic_key, score)
        self.skill_signals[topic_key] = round((1 - alpha) * prev + alpha * score, 2)
