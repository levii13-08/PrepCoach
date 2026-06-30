"""Phase 8 skill gap analysis."""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from app.data_prep.normalization import extract_skills
from app.schemas.data_models import JobDescription
from app.schemas.interview_state import InterviewState, RoundRecord

WEAK_THRESHOLD = 6.0
STRONG_THRESHOLD = 7.5

_SKILL_ALIASES = {
    "machine learning": "ml",
    "ml": "ml",
    "deep learning": "deep_learning",
    "neural networks": "deep_learning",
    "mlops": "mlops",
    "software engineering": "swe",
    "swe": "swe",
    "data structures": "dsa",
    "algorithms": "dsa",
    "dsa": "dsa",
    "object oriented programming": "oop",
    "oop": "oop",
    "system design": "system_design",
    "distributed systems": "system_design",
    "artificial intelligence": "ai",
    "ai": "ai",
    "large language models": "llms",
    "llms": "llms",
    "llm": "llms",
    "rag": "rag",
    "retrieval augmented generation": "rag",
    "agents": "agents",
    "langgraph": "agents",
    "python": "swe",
    "sql": "swe",
    "pytorch": "deep_learning",
    "transformers": "llms",
    "vector databases": "rag",
}


class SkillEvidence(BaseModel):
    source: str
    detail: str
    citation: Optional[str] = None
    score: Optional[float] = None
    doc_ids: list[str] = Field(default_factory=list)


class SkillGap(BaseModel):
    skill: str
    signal_key: str
    status: str
    score: Optional[float] = None
    evidence: list[SkillEvidence] = Field(default_factory=list)


class SkillGapReport(BaseModel):
    role: str
    missing: list[SkillGap] = Field(default_factory=list)
    weak: list[SkillGap] = Field(default_factory=list)
    developing: list[SkillGap] = Field(default_factory=list)
    strong: list[SkillGap] = Field(default_factory=list)


def normalize_skill(skill: str) -> str:
    key = re.sub(r"[^a-z0-9]+", " ", skill.lower()).strip()
    return _SKILL_ALIASES.get(key, key.replace(" ", "_"))


def _skill_terms(skill: str, signal_key: str) -> set[str]:
    normalized_skill = re.sub(r"[^a-z0-9]+", " ", skill.lower()).strip()
    normalized_signal = signal_key.replace("_", " ").strip()
    terms = {
        normalized_skill,
        normalized_signal,
        skill.lower().strip(),
        signal_key.lower().strip(),
    }

    for alias, canonical in _SKILL_ALIASES.items():
        if canonical == signal_key:
            terms.add(alias)

    return {term for term in terms if term}


def _job_role(job: JobDescription | dict[str, Any]) -> str:
    if isinstance(job, JobDescription):
        return job.role.value
    return str(job.get("role") or job.get("role_title") or "unknown")


def _job_skills(job: JobDescription | dict[str, Any]) -> list[str]:
    if isinstance(job, JobDescription):
        return job.skills
    skills = job.get("skills", [])
    if isinstance(skills, str):
        return [part.strip() for part in re.split(r"[,|;/]", skills) if part.strip()]
    if skills:
        return [str(skill) for skill in skills]
    fallback_text = " ".join([str(job.get("role_title") or ""), str(job.get("requirements") or "")]).strip()
    return extract_skills(fallback_text)


def _job_requirements(job: JobDescription | dict[str, Any]) -> str:
    if isinstance(job, JobDescription):
        return job.requirements
    return str(job.get("requirements") or "")


def _job_source(job: JobDescription | dict[str, Any]) -> tuple[str, Optional[str]]:
    if isinstance(job, JobDescription):
        return job.source, job.source_url
    return str(job.get("source") or "job_description"), job.get("source_url")


def _requirement_excerpt(requirements: str, terms: set[str]) -> Optional[str]:
    for chunk in re.split(r"(?<=[.!?])\s+|\n+", requirements):
        snippet = chunk.strip()
        lowered = snippet.lower()
        if snippet and any(term in lowered for term in terms):
            return snippet
    return None


def _job_evidence(
    job: JobDescription | dict[str, Any],
    skill: str,
    signal_key: str,
) -> list[SkillEvidence]:
    requirements = _job_requirements(job)
    source, source_url = _job_source(job)
    terms = _skill_terms(skill, signal_key)
    excerpt = _requirement_excerpt(requirements, terms)
    detail = excerpt or f"Job description lists {skill} as a target skill."
    citation = source_url or source
    return [
        SkillEvidence(
            source="job_description",
            detail=detail,
            citation=citation,
        )
    ]


def _history_evidence(skill: str, signal_key: str, history: Iterable[RoundRecord]) -> list[SkillEvidence]:
    terms = _skill_terms(skill, signal_key)
    evidence = []
    for record in history:
        text = " ".join([record.topic or "", record.question, " ".join(record.expected_points)]).lower()
        if not any(term in text for term in terms):
            continue
        citation_parts = [f"round {record.round}"]
        if record.retrieved_doc_ids:
            citation_parts.append(", ".join(record.retrieved_doc_ids))
        evidence.append(
            SkillEvidence(
                source=f"round:{record.round}",
                detail=record.question,
                citation=" | ".join(citation_parts),
                score=record.scores.total,
                doc_ids=record.retrieved_doc_ids,
            )
        )
    return evidence


def _signal_evidence(signal_key: str, score: Optional[float]) -> list[SkillEvidence]:
    if score is None:
        return []
    return [
        SkillEvidence(
            source="skill_signals",
            detail=f"Rolling interview score for {signal_key}",
            citation=f"skill_signals[{signal_key}]",
            score=score,
        )
    ]


def analyze_skill_gaps(
    job: JobDescription | dict[str, Any],
    skill_signals: dict[str, float],
    history: Optional[list[RoundRecord]] = None,
) -> SkillGapReport:
    history = history or []
    report = SkillGapReport(role=_job_role(job))

    seen = set()
    for skill in _job_skills(job):
        signal_key = normalize_skill(skill)
        if signal_key in seen:
            continue
        seen.add(signal_key)

        score = skill_signals.get(signal_key)
        evidence = (
            _job_evidence(job, skill, signal_key)
            + _signal_evidence(signal_key, score)
            + _history_evidence(skill, signal_key, history)
        )
        if score is None:
            gap = SkillGap(skill=skill, signal_key=signal_key, status="missing", evidence=evidence)
            report.missing.append(gap)
        elif score < WEAK_THRESHOLD:
            gap = SkillGap(skill=skill, signal_key=signal_key, status="weak", score=score, evidence=evidence)
            report.weak.append(gap)
        elif score >= STRONG_THRESHOLD:
            gap = SkillGap(skill=skill, signal_key=signal_key, status="strong", score=score, evidence=evidence)
            report.strong.append(gap)
        else:
            gap = SkillGap(skill=skill, signal_key=signal_key, status="developing", score=score, evidence=evidence)
            report.developing.append(gap)

    return report


def analyze_state_against_job(
    state: InterviewState,
    job: JobDescription | dict[str, Any],
) -> SkillGapReport:
    return analyze_skill_gaps(job, state.skill_signals, state.history)
