"""Phase 4: Structured interview question generation with validation + retry."""

import json
import uuid
from typing import Callable, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.config import (
    COLLECTION_TECH_DOCS,
    DEFAULT_DIFFICULTY,
    RAG_RETRIEVE_K,
    RAG_TOP_K,
    RAG_USE_HYBRID,
    RAG_USE_MULTI_QUERY,
    RAG_USE_RERANK,
)
from app.llm.groq_llm import generate
from app.rag.retriever import format_context, retrieve
from app.schemas.data_models import Difficulty, InterviewQuestion, RoleCode, Topic

_MAX_RETRIES = 3
_MAX_CONTEXT_CHARS = 6000


class QuestionGenerationRequest(BaseModel):
    role: str = Field(min_length=2)
    num_questions: int = Field(default=5, ge=1, le=20)
    topic: Optional[Topic] = None
    difficulty: Difficulty = Difficulty(DEFAULT_DIFFICULTY)
    collection: str = COLLECTION_TECH_DOCS
    use_multi_query: Optional[bool] = None
    use_hybrid: Optional[bool] = None
    use_rerank: Optional[bool] = None

    @field_validator("role")
    @classmethod
    def normalize_role(cls, value: str) -> str:
        return " ".join(value.strip().split())


class QuestionGenerationResponse(BaseModel):
    role: str
    difficulty: Difficulty
    topic: Optional[Topic] = None
    questions: list[InterviewQuestion]
    retrieval_warning: Optional[str] = None
    parse_attempts: int = Field(ge=1)


_JSON_GEN_TEMPLATE = """You are an expert interview question generator for {role} positions.

Use the following context from job descriptions, technical documentation, and existing interview questions to generate relevant, grounded interview questions.

Context:
{context}

Generate exactly {num_questions} interview questions for a {role} position.
Topic: {topic}
Difficulty: {difficulty}

Return ONLY a valid JSON array of objects. Do not wrap it in markdown.
Each object must have these exact fields:
- "question": string (min 10 characters)
- "topic": one of {topic_values}
- "difficulty": "easy", "medium", or "hard"
- "role_tags": array of strings from {role_values}
- "question_type": one of "theory", "coding", "system_design", "behavioral", "ml", "swe"
- "expected_points": array of strings, each a key point for the ideal answer
- "source": string describing which context document inspired this question

Requirements:
- Match the requested role, topic, and difficulty.
- Make every question specific to the skills, tools, and technologies in the context.
- Use role_tags values from the allowed enum only.
- Include at least 2 expected_points for each question.
- If context is thin, generate a broadly relevant question and set source to "fallback: insufficient retrieved context".

JSON:"""


def _build_json_prompt(
    role: str,
    context: str,
    num_questions: int,
    topic: Optional[str],
    difficulty: str,
    previous_error: Optional[str] = None,
) -> str:
    prompt = _JSON_GEN_TEMPLATE.format(
        role=role,
        context=context,
        num_questions=num_questions,
        topic=topic or "any",
        difficulty=difficulty,
        topic_values=json.dumps([topic.value for topic in Topic]),
        role_values=json.dumps([role.value for role in RoleCode]),
    )

    if previous_error:
        prompt += (
            "\n\nYour previous response had a parsing or validation error:\n"
            f"{previous_error}\n\nFix the JSON and try again. Return valid JSON only."
        )

    return prompt


def _parse_questions(raw: str) -> list[dict]:
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
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(raw[start : end + 1])

    if isinstance(data, dict):
        data = data.get("questions", data.get("interview_questions", [data]))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")

    return data


def _validate_questions(data: list[dict]) -> list[InterviewQuestion]:
    questions = []
    for item in data:
        item.setdefault("id", str(uuid.uuid4()))
        questions.append(InterviewQuestion(**item))
    return questions


def _build_retrieval_query(request: QuestionGenerationRequest) -> str:
    parts = [request.role, "interview question", request.difficulty.value]
    if request.topic:
        parts.append(request.topic.value)
    return " ".join(parts)


def _format_context_or_warning(results: list[dict]) -> tuple[str, Optional[str]]:
    if not results:
        return (
            "No retrieved context was available.",
            "No retrieved context was available; generated questions may be generic.",
        )

    context = format_context(results)
    return context[:_MAX_CONTEXT_CHARS], None


def generate_question_set(
    request: QuestionGenerationRequest,
    llm_generate: Callable[[str], str] = generate,
) -> QuestionGenerationResponse:
    results = retrieve(
        _build_retrieval_query(request),
        collection=request.collection,
        k=RAG_TOP_K,
        retrieve_k=RAG_RETRIEVE_K,
        use_multi_query=request.use_multi_query if request.use_multi_query is not None else RAG_USE_MULTI_QUERY,
        use_hybrid=request.use_hybrid if request.use_hybrid is not None else RAG_USE_HYBRID,
        use_rerank=request.use_rerank if request.use_rerank is not None else RAG_USE_RERANK,
    )
    context, retrieval_warning = _format_context_or_warning(results)

    last_error = None
    for attempt in range(1, _MAX_RETRIES + 1):
        prompt = _build_json_prompt(
            role=request.role,
            context=context,
            num_questions=request.num_questions,
            topic=request.topic.value if request.topic else None,
            difficulty=request.difficulty.value,
            previous_error=last_error,
        )
        raw = llm_generate(prompt)

        try:
            questions = _validate_questions(_parse_questions(raw))
            if len(questions) != request.num_questions:
                raise ValueError(
                    f"Expected {request.num_questions} questions, got {len(questions)}"
                )
            return QuestionGenerationResponse(
                role=request.role,
                difficulty=request.difficulty,
                topic=request.topic,
                questions=questions,
                retrieval_warning=retrieval_warning,
                parse_attempts=attempt,
            )
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    raise RuntimeError(
        f"Failed to generate valid questions after {_MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def generate_questions(
    role: str,
    num_questions: int = 5,
    topic: Optional[str] = None,
    difficulty: Optional[str] = None,
    collection: str = COLLECTION_TECH_DOCS,
    use_multi_query: Optional[bool] = None,
    use_hybrid: Optional[bool] = None,
    use_rerank: Optional[bool] = None,
) -> list[InterviewQuestion]:
    request = QuestionGenerationRequest(
        role=role,
        num_questions=num_questions,
        topic=topic,
        difficulty=difficulty or DEFAULT_DIFFICULTY,
        collection=collection,
        use_multi_query=use_multi_query,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
    )
    return generate_question_set(request).questions
