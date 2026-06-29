"""Phase 7 adaptive planner rules."""

from app.schemas.data_models import Difficulty, Topic
from app.schemas.interview_state import InterviewState, RoundType, SessionStatus

TOPIC_SEQUENCE = [
    Topic.ML,
    Topic.RAG,
    Topic.SYSTEM_DESIGN,
    Topic.SWE,
    Topic.DSA,
]


def decrease_difficulty(difficulty: Difficulty) -> Difficulty:
    order = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
    return order[max(0, order.index(difficulty) - 1)]


def increase_difficulty(difficulty: Difficulty) -> Difficulty:
    order = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
    return order[min(len(order) - 1, order.index(difficulty) + 1)]


def round_type_for_topic(topic: Topic) -> RoundType:
    if topic == Topic.SYSTEM_DESIGN:
        return RoundType.SYSTEM_DESIGN
    if topic in {Topic.DSA, Topic.OOP}:
        return RoundType.CODING
    if topic == Topic.SWE:
        return RoundType.SWE
    return RoundType.ML


def weakest_topic(state: InterviewState) -> Topic | None:
    weak_topics = [
        (topic, score)
        for topic, score in state.skill_signals.items()
        if score < 6 and topic in Topic._value2member_map_
    ]
    if not weak_topics:
        return None
    topic, _ = min(weak_topics, key=lambda item: item[1])
    return Topic(topic)


def next_sequence_topic(state: InterviewState) -> Topic:
    return TOPIC_SEQUENCE[state.current_round % len(TOPIC_SEQUENCE)]


def plan_next_round(state: InterviewState) -> InterviewState:
    if state.current_round >= state.max_rounds:
        state.session_status = SessionStatus.COMPLETED
        return state

    latest_score = state.history[-1].scores.total if state.history else None

    if latest_score is not None and latest_score < 5:
        state.difficulty = decrease_difficulty(state.difficulty)
    elif latest_score is not None and latest_score >= 8:
        state.difficulty = increase_difficulty(state.difficulty)

    if state.flags.get("fixed_topic") and state.topic:
        next_topic = state.topic
    else:
        next_topic = weakest_topic(state) or next_sequence_topic(state)

    state.topic = next_topic
    state.round_type = round_type_for_topic(next_topic)
    state.user_answer = None
    state.question = None
    state.expected_points = []
    state.retrieved_context = []
    state.scores = None
    state.flags["awaiting_answer"] = False
    return state
