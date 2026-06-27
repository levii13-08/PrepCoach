"""Phase 5-7 LangGraph workflow for adaptive mock interviews."""

from __future__ import annotations

from typing import Any, Literal

from app.agents.planner import plan_next_round
from app.config import COLLECTION_TECH_DOCS, RAG_RETRIEVE_K, RAG_TOP_K
from app.evaluation.evaluator import EvaluationRequest, evaluate_answer, heuristic_evaluate_answer
from app.generation.question_generator import QuestionGenerationRequest, generate_question_set
from app.rag.retriever import retrieve
from app.schemas.data_models import RoleCode
from app.schemas.interview_state import (
    ContextChunk,
    InterviewState,
    RoundRecord,
    SessionStatus,
)

try:
    from langgraph.graph import END, StateGraph
except ImportError:
    END = "__end__"
    StateGraph = None

GraphState = dict[str, Any]


def _state(data: GraphState) -> InterviewState:
    return data if isinstance(data, InterviewState) else InterviewState.model_validate(data)


def _dump(state: InterviewState) -> GraphState:
    return state.model_dump(mode="python")


def _role_label(role: RoleCode) -> str:
    labels = {
        RoleCode.MLE: "Machine Learning Engineer",
        RoleCode.AI: "AI Engineer",
        RoleCode.DS: "Data Scientist",
        RoleCode.SWE: "Software Engineer",
        RoleCode.GENAI: "Generative AI Engineer",
    }
    return labels.get(role, role.value)


def planner_node(data: GraphState) -> GraphState:
    state = _state(data)
    return _dump(plan_next_round(state))


def retriever_node(data: GraphState) -> GraphState:
    state = _state(data)
    if state.flags.get("mock_llm", False):
        state.retrieved_context = []
        state.flags["retrieval_warning"] = "Retrieval skipped for mock LLM mode."
        return _dump(state)

    query = f"{_role_label(state.role)} {state.topic.value if state.topic else ''} {state.difficulty.value} interview question"

    try:
        results = retrieve(
            query,
            collection=COLLECTION_TECH_DOCS,
            k=RAG_TOP_K,
            retrieve_k=RAG_RETRIEVE_K,
            use_multi_query=False,
            use_hybrid=False,
            use_rerank=False,
        )
    except Exception as exc:
        state.flags["retrieval_error"] = f"{type(exc).__name__}: {exc}"
        results = []

    state.retrieved_context = [
        ContextChunk(
            content=result.get("content", ""),
            collection=result.get("collection", COLLECTION_TECH_DOCS),
            doc_id=str(
                result.get("metadata", {}).get("id")
                or result.get("metadata", {}).get("filename")
                or index
            ),
            score=float(result.get("score", 0.0)),
            metadata=result.get("metadata", {}),
        )
        for index, result in enumerate(results)
    ]
    return _dump(state)


def _mock_question(state: InterviewState) -> tuple[str, list[str], str]:
    topic = state.topic.value if state.topic else "general"
    question = (
        f"For a {_role_label(state.role)} role, explain how you would approach a "
        f"{topic} problem at {state.difficulty.value} difficulty."
    )
    expected_points = [
        "state assumptions and tradeoffs",
        "describe the core technical approach",
        "connect the answer to production constraints",
    ]
    return question, expected_points, "mock: phase 5 deterministic interviewer"


def interviewer_node(data: GraphState) -> GraphState:
    state = _state(data)

    if state.flags.get("mock_llm", False):
        question, expected_points, source = _mock_question(state)
        state.question = question
        state.expected_points = expected_points
        state.flags["question_source"] = source
        state.flags["awaiting_answer"] = True
        return _dump(state)

    try:
        response = generate_question_set(
            QuestionGenerationRequest(
                role=_role_label(state.role),
                num_questions=1,
                topic=state.topic,
                difficulty=state.difficulty,
                collection=COLLECTION_TECH_DOCS,
                use_multi_query=False,
                use_hybrid=False,
                use_rerank=False,
            )
        )
        question = response.questions[0]
        state.question = question.question
        state.expected_points = question.expected_points
        state.flags["question_source"] = question.source
        if response.retrieval_warning:
            state.flags["retrieval_warning"] = response.retrieval_warning
    except Exception as exc:
        question, expected_points, source = _mock_question(state)
        state.question = question
        state.expected_points = expected_points
        state.flags["question_source"] = source
        state.flags["generation_error"] = f"{type(exc).__name__}: {exc}"

    state.flags["awaiting_answer"] = True
    return _dump(state)


def wait_answer_node(data: GraphState) -> GraphState:
    state = _state(data)
    state.flags["awaiting_answer"] = True
    return _dump(state)


def evaluator_node(data: GraphState) -> GraphState:
    state = _state(data)
    answer = (state.user_answer or "").strip()
    request = EvaluationRequest(
        question=state.question or "",
        expected_points=state.expected_points,
        user_answer=answer,
        role=_role_label(state.role),
        topic=state.topic.value if state.topic else None,
        difficulty=state.difficulty.value,
        retrieved_context=[chunk.content for chunk in state.retrieved_context],
    )

    if state.flags.get("mock_llm", False):
        state.scores = heuristic_evaluate_answer(request)
    else:
        try:
            state.scores = evaluate_answer(request)
        except Exception as exc:
            state.scores = heuristic_evaluate_answer(request)
            state.flags["evaluation_error"] = f"{type(exc).__name__}: {exc}"

    state.current_round += 1
    topic_key = state.topic.value if state.topic else state.round_type.value
    state.update_skill_signal(topic_key, state.scores.total)
    state.history.append(
        RoundRecord(
            round=state.current_round,
            round_type=state.round_type,
            topic=state.topic.value if state.topic else None,
            difficulty=state.difficulty,
            question=state.question or "",
            user_answer=answer,
            expected_points=state.expected_points,
            scores=state.scores,
            retrieved_doc_ids=[chunk.doc_id for chunk in state.retrieved_context],
        )
    )
    state.flags["awaiting_answer"] = False
    if state.current_round >= state.max_rounds:
        state.session_status = SessionStatus.COMPLETED
    return _dump(state)


def route_after_planner(data: GraphState) -> Literal["retriever", "__end__"]:
    state = _state(data)
    return "__end__" if state.session_status == SessionStatus.COMPLETED else "retriever"


def route_after_wait_answer(data: GraphState) -> Literal["evaluator", "__end__"]:
    state = _state(data)
    return "evaluator" if state.user_answer else "__end__"


def build_graph():
    if StateGraph is None:
        raise RuntimeError("langgraph is not installed. Run `pip install -r requirements.txt`.")

    graph = StateGraph(GraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("interviewer", interviewer_node)
    graph.add_node("wait_answer", wait_answer_node)
    graph.add_node("evaluator", evaluator_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", route_after_planner, {"retriever": "retriever", "__end__": END})
    graph.add_edge("retriever", "interviewer")
    graph.add_edge("interviewer", "wait_answer")
    graph.add_conditional_edges("wait_answer", route_after_wait_answer, {"evaluator": "evaluator", "__end__": END})
    graph.add_edge("evaluator", "planner")
    return graph.compile()


def start_round(state: InterviewState) -> InterviewState:
    data = planner_node(_dump(state))
    if route_after_planner(data) == "__end__":
        return _state(data)
    data = retriever_node(data)
    data = interviewer_node(data)
    data = wait_answer_node(data)
    return _state(data)


def submit_answer(state: InterviewState, answer: str) -> InterviewState:
    state.user_answer = answer
    return _state(evaluator_node(_dump(state)))
