"""Evaluation output — Phase 6; schema defined in Phase 0."""

from pydantic import BaseModel, Field, model_validator

from app.config import RUBRIC_WEIGHTS


class EvaluationScore(BaseModel):
    correctness: float = Field(ge=0, le=10)
    completeness: float = Field(ge=0, le=10)
    clarity: float = Field(ge=0, le=10)
    alignment: float = Field(ge=0, le=10)
    total: float = Field(ge=0, le=10)
    feedback: str = Field(min_length=1)

    @model_validator(mode="after")
    def sync_total(self) -> "EvaluationScore":
        expected = compute_total_score(self)
        if abs(self.total - expected) > 0.15:
            object.__setattr__(self, "total", round(expected, 2))
        return self


def compute_total_score(
    scores: EvaluationScore | dict,
) -> float:
    if isinstance(scores, EvaluationScore):
        c, comp, cl, a = scores.correctness, scores.completeness, scores.clarity, scores.alignment
    else:
        c = scores["correctness"]
        comp = scores["completeness"]
        cl = scores["clarity"]
        a = scores["alignment"]
    return round(
        RUBRIC_WEIGHTS["correctness"] * c
        + RUBRIC_WEIGHTS["completeness"] * comp
        + RUBRIC_WEIGHTS["clarity"] * cl
        + RUBRIC_WEIGHTS["alignment"] * a,
        2,
    )
