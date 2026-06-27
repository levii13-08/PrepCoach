"""Shared interview orchestration for CLI, API, and UI surfaces."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.agents.graph import start_round, submit_answer
from app.coaching.coach import build_learning_roadmap
from app.coaching.skill_gap import analyze_state_against_job
from app.data_access import find_job_by_id, find_matching_job
from app.llm.groq_llm import has_valid_groq_key
from app.memory.db import load_latest_session, load_session, save_session
from app.schemas.data_models import Difficulty, RoleCode, Topic
from app.schemas.interview_state import InterviewState, SessionStatus


class StartInterviewRequest(BaseModel):
    role: RoleCode = RoleCode.MLE
    user_id: Optional[str] = None
    max_rounds: int = Field(default=3, ge=1, le=10)
    difficulty: Difficulty = Difficulty.MEDIUM
    topic: Optional[Topic] = None
    target_job_id: Optional[str] = None
    dataset: str = Field(default="demo", pattern="^(light|demo|full)$")
    mock_llm: bool = True
    db_path: Optional[str] = None


class AnswerInterviewRequest(BaseModel):
    answer: str = Field(min_length=1)
    dataset: str = Field(default="demo", pattern="^(light|demo|full)$")
    db_path: Optional[str] = None


def _enrich_state(state: InterviewState, dataset: str, db_path: Optional[str]) -> InterviewState:
    target_job = None
    if state.target_job_id:
        target_job = find_job_by_id(state.target_job_id, dataset=dataset)
    if target_job is None:
        target_job = find_matching_job(state.role, dataset=dataset)

    if target_job is not None and state.history:
        report = analyze_state_against_job(state, target_job)
        roadmap = build_learning_roadmap(report)
        state.skill_gap_report = report.model_dump()
        state.learning_roadmap = roadmap.model_dump()
        state.coach_feedback = roadmap.summary
        save_session(state, db_path=db_path)
    return state


def begin_interview(request: StartInterviewRequest) -> InterviewState:
    if not request.mock_llm and not has_valid_groq_key():
        raise ValueError("Missing valid GROQ_API_KEY in .env. Use mock mode or add a real Groq key.")

    state = InterviewState(
        role=request.role,
        user_id=request.user_id,
        target_job_id=request.target_job_id,
        max_rounds=request.max_rounds,
        difficulty=request.difficulty,
        topic=request.topic,
        flags={"mock_llm": request.mock_llm, "fixed_topic": bool(request.topic)},
    )
    save_session(state, db_path=request.db_path)
    state = start_round(state)
    save_session(state, db_path=request.db_path)
    return _enrich_state(state, request.dataset, request.db_path)


def answer_interview(
    session_id: str,
    request: AnswerInterviewRequest,
) -> InterviewState:
    state = load_session(session_id, db_path=request.db_path)
    if state.session_status == SessionStatus.COMPLETED:
        return _enrich_state(state, request.dataset, request.db_path)

    state = submit_answer(state, request.answer)
    save_session(state, db_path=request.db_path)
    if state.session_status != SessionStatus.COMPLETED:
        state = start_round(state)
        save_session(state, db_path=request.db_path)
    return _enrich_state(state, request.dataset, request.db_path)


def get_session(session_id: str, db_path: Optional[str] = None) -> InterviewState:
    return load_session(session_id, db_path=db_path)


def get_latest_session(user_id: Optional[str], db_path: Optional[str] = None) -> InterviewState:
    return load_latest_session(user_id=user_id, db_path=db_path)
