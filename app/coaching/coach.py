"""Phase 9 coaching roadmap generation."""

from __future__ import annotations

from typing import Callable, Optional

from pydantic import BaseModel, Field

from app.config import COLLECTION_TECH_DOCS
from app.coaching.skill_gap import SkillGap, SkillGapReport
from app.rag.retriever import retrieve


class CoachingResource(BaseModel):
    collection: str = COLLECTION_TECH_DOCS
    title: str
    source: str
    doc_id: str
    snippet: str


class RoadmapItem(BaseModel):
    skill: str
    status: str
    priority: int = Field(ge=1, le=5)
    goal: str
    why_this_matters: str
    practice_tasks: list[str] = Field(default_factory=list)
    resources: list[CoachingResource] = Field(default_factory=list)
    evidence_summary: list[str] = Field(default_factory=list)


class LearningRoadmap(BaseModel):
    role: str
    summary: str
    items: list[RoadmapItem] = Field(default_factory=list)


Retriever = Callable[[str], list[dict]]


def _default_resource_retriever(query: str) -> list[dict]:
    try:
        return retrieve(
            query,
            collection=COLLECTION_TECH_DOCS,
            k=3,
            retrieve_k=5,
            use_multi_query=False,
            use_hybrid=False,
            use_rerank=False,
        )
    except Exception:
        return []


def _resources_for_gap(gap: SkillGap, resource_retriever: Retriever) -> list[CoachingResource]:
    resources = []
    for result in resource_retriever(gap.signal_key):
        metadata = result.get("metadata", {})
        collection = str(result.get("collection") or metadata.get("collection") or COLLECTION_TECH_DOCS)
        if collection != COLLECTION_TECH_DOCS:
            continue
        doc_id = str(metadata.get("doc_id") or metadata.get("filename") or result.get("source") or "unknown")
        resources.append(
            CoachingResource(
                collection=collection,
                title=str(metadata.get("title") or metadata.get("filename") or gap.skill),
                source=str(metadata.get("source") or result.get("source") or COLLECTION_TECH_DOCS),
                doc_id=doc_id,
                snippet=result.get("content", "")[:240],
            )
        )
    return resources


def _evidence_summary(gap: SkillGap) -> list[str]:
    summary: list[str] = []
    for evidence in gap.evidence[:3]:
        detail = evidence.detail.strip()
        if len(detail) > 120:
            detail = detail[:117].rstrip() + "..."
        if evidence.score is not None:
            summary.append(f"{evidence.source}: {detail} (score {evidence.score}/10)")
        else:
            summary.append(f"{evidence.source}: {detail}")
    return summary


def _resource_tasks(skill: str, signal_key: str) -> list[str]:
    normalized = signal_key.lower()
    if normalized in {"rag", "agents", "llms"}:
        return [
            f"Draw the end-to-end {skill} pipeline on paper.",
            f"Explain retrieval, ranking, and trust tradeoffs for {skill}.",
            f"Practice one system-style interview answer on {skill}.",
        ]
    if normalized in {"ml", "deep_learning", "mlops"}:
        return [
            f"Build a baseline explanation for {skill} with metrics and failure modes.",
            f"Practice one production-focused {skill} question out loud.",
            "Tie model choices back to latency, drift, and deployment constraints.",
        ]
    if normalized in {"dsa", "oop", "swe"}:
        return [
            f"Review the core patterns behind {skill}.",
            f"Solve one medium practice problem for {skill}.",
            "Explain complexity, edge cases, and testing strategy clearly.",
        ]
    return [
        f"Review the core concepts for {skill}.",
        f"Practice one interview answer focused on {skill}.",
        "Summarize the strongest tradeoffs and pitfalls in your own words.",
    ]


def _why_this_matters(gap: SkillGap) -> str:
    for evidence in gap.evidence:
        if evidence.source == "job_description":
            return evidence.detail
    if gap.score is not None:
        return f"Your current signal for {gap.skill} is {gap.score}/10, so it is limiting interview readiness."
    return f"{gap.skill} appears in the target role and needs stronger interview coverage."


def _item_for_gap(gap: SkillGap, priority: int, resource_retriever: Retriever) -> RoadmapItem:
    if gap.status == "missing":
        goal = f"Build first-pass working knowledge of {gap.skill}."
    elif gap.status == "weak":
        goal = f"Raise {gap.skill} from weak to interview-ready."
    else:
        goal = f"Maintain and sharpen {gap.skill}."
    tasks = _resource_tasks(gap.skill, gap.signal_key)
    if gap.status == "weak":
        tasks.insert(0, f"Redo the weakest {gap.skill} round and cover every expected point.")
    elif gap.status == "missing":
        tasks.insert(0, f"Start with a 20-minute primer on {gap.skill} before practice.")
    else:
        tasks.insert(0, f"Practice one harder follow-up on {gap.skill}.")

    return RoadmapItem(
        skill=gap.skill,
        status=gap.status,
        priority=priority,
        goal=goal,
        why_this_matters=_why_this_matters(gap),
        practice_tasks=tasks,
        resources=_resources_for_gap(gap, resource_retriever),
        evidence_summary=_evidence_summary(gap),
    )


def build_learning_roadmap(
    report: SkillGapReport,
    resource_retriever: Optional[Retriever] = None,
) -> LearningRoadmap:
    retriever_fn = resource_retriever or _default_resource_retriever
    focus_gaps = report.missing + report.weak + report.developing
    if not focus_gaps:
        focus_gaps = report.strong[:3]

    items = [
        _item_for_gap(gap, priority=min(index + 1, 5), resource_retriever=retriever_fn)
        for index, gap in enumerate(focus_gaps[:5])
    ]

    missing_count = len(report.missing)
    weak_count = len(report.weak)
    summary = (
        f"{report.role} roadmap: {missing_count} missing skills and "
        f"{weak_count} weak skills need attention first. "
        f"Resources and tasks are prioritized from the target job requirements, interview signals, "
        f"and the {COLLECTION_TECH_DOCS} collection."
    )
    if not items:
        summary = f"{report.role} roadmap: no job skills were available to analyze."

    return LearningRoadmap(role=report.role, summary=summary, items=items)
