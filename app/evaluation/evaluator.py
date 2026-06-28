"""Phase 6 rubric-based answer evaluation."""

import json
from typing import Callable, Optional

from pydantic import BaseModel, Field, ValidationError

from app.config import LLM_TEMPERATURE_EVAL, RUBRIC_WEIGHTS
from app.llm.groq_llm import generate_eval
from app.schemas.evaluation import EvaluationScore

_MAX_RETRIES = 3
_MAX_CONTEXT_CHARS = 5000


class EvaluationRequest(BaseModel):
    question: str = Field(min_length=1)
    expected_points: list[str] = Field(default_factory=list)
    user_answer: str = Field(min_length=1)
    role: str = Field(min_length=1)
    topic: Optional[str] = None
    difficulty: Optional[str] = None
    retrieved_context: list[str] = Field(default_factory=list)


_EVALUATION_TEMPLATE = """You are a strict interview evaluator.

Evaluate the candidate answer using this rubric:
- correctness: 40%, factual and technical accuracy
- completeness: 25%, coverage of expected points and important concepts
- clarity: 15%, structure, precision, and communication quality
- alignment: 20%, fit for the role, topic, and question intent

Role: {role}
Topic: {topic}
Difficulty: {difficulty}

Question:
{question}

Expected points:
{expected_points}

Retrieved reference context:
{retrieved_context}

Candidate answer:
{user_answer}

Return ONLY valid JSON with these exact keys:
{{
  "correctness": number from 0 to 10,
  "completeness": number from 0 to 10,
  "clarity": number from 0 to 10,
  "alignment": number from 0 to 10,
  "total": weighted total from 0 to 10,
  "feedback": "one concise paragraph with the strongest fix"
}}

Use temperature {temperature}. Be calibrated: short vague answers should score below 5.
JSON:"""


def _format_context(context: list[str]) -> str:
    if not context:
        return "No retrieved reference context was provided."
    return "\n\n---\n\n".join(context)[:_MAX_CONTEXT_CHARS]


def build_evaluation_prompt(
    request: EvaluationRequest,
    previous_error: Optional[str] = None,
) -> str:
    prompt = _EVALUATION_TEMPLATE.format(
        role=request.role,
        topic=request.topic or "general",
        difficulty=request.difficulty or "unspecified",
        question=request.question,
        expected_points=json.dumps(request.expected_points, indent=2),
        retrieved_context=_format_context(request.retrieved_context),
        user_answer=request.user_answer,
        temperature=LLM_TEMPERATURE_EVAL,
    )
    if previous_error:
        prompt += (
            "\n\nYour previous response failed JSON parsing or schema validation:\n"
            f"{previous_error}\nReturn corrected JSON only."
        )
    return prompt


def _parse_score(raw: str) -> EvaluationScore:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(raw[start : end + 1])

    return EvaluationScore(**data)


def evaluate_answer(
    request: EvaluationRequest,
    llm_generate: Callable[[str], str] = generate_eval,
) -> EvaluationScore:
    last_error = None
    for _ in range(_MAX_RETRIES):
        prompt = build_evaluation_prompt(request, previous_error=last_error)
        try:
            return _parse_score(llm_generate(prompt))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    raise RuntimeError(
        f"Failed to generate valid evaluation after {_MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def heuristic_evaluate_answer(request: EvaluationRequest) -> EvaluationScore:
    answer = request.user_answer.strip()
    token_count = len(answer.split())
    answer_lower = answer.lower()
    coverage = sum(
        1
        for point in request.expected_points
        if any(word.lower().strip(".,:;()") in answer_lower for word in point.split())
    )
    coverage_ratio = coverage / max(len(request.expected_points), 1)

    correctness = min(10.0, 2.0 + token_count / 18 + coverage_ratio * 4)
    completeness = min(10.0, 2.0 + token_count / 16 + coverage_ratio * 5)
    clarity = min(10.0, 3.0 + token_count / 24)
    alignment = min(10.0, 3.0 + coverage_ratio * 5)

    total = round(
        RUBRIC_WEIGHTS["correctness"] * correctness
        + RUBRIC_WEIGHTS["completeness"] * completeness
        + RUBRIC_WEIGHTS["clarity"] * clarity
        + RUBRIC_WEIGHTS["alignment"] * alignment,
        2,
    )

    return EvaluationScore(
        correctness=round(correctness, 2),
        completeness=round(completeness, 2),
        clarity=round(clarity, 2),
        alignment=round(alignment, 2),
        total=total,
        feedback="Heuristic score for local smoke tests. Use the LLM evaluator for calibrated Phase 6 scoring.",
    )
