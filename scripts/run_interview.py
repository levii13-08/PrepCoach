"""Run a Phase 5 mock interview in the terminal."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.graph import start_round, submit_answer
from app.coaching.coach import build_learning_roadmap
from app.coaching.skill_gap import analyze_state_against_job
from app.data_access import find_job_by_id, find_matching_job
from app.llm.groq_llm import has_valid_groq_key
from app.memory.db import load_latest_session, load_session, save_session
from app.schemas.data_models import Difficulty, RoleCode, Topic
from app.schemas.interview_state import InterviewState, SessionStatus


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CLI mock interview.")
    parser.add_argument("--role", choices=[role.value for role in RoleCode], default=RoleCode.MLE.value)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--difficulty", choices=[difficulty.value for difficulty in Difficulty], default=Difficulty.MEDIUM.value)
    parser.add_argument("--topic", choices=[topic.value for topic in Topic], default=None)
    parser.add_argument("--target-job-id", default=None, help="Normalized job id used for skill-gap and coaching output.")
    parser.add_argument("--dataset", choices=["full", "light", "demo"], default="demo", help="Which normalized dataset profile to read jobs from.")
    parser.add_argument("--db-path", default=None, help="SQLite path for saved interview sessions.")
    parser.add_argument("--resume-session", default=None, help="Resume a saved session by session_id.")
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume the latest saved active session for the given --user-id.",
    )
    parser.add_argument(
        "--mock-llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic local questions instead of calling the LLM.",
    )
    parser.add_argument(
        "--auto-answer",
        action="store_true",
        help="Use canned answers so the 3-round smoke test can run unattended.",
    )
    return parser.parse_args()


def _read_answer(auto_answer: bool, round_number: int) -> str:
    if auto_answer:
        return (
            "I would clarify requirements, describe the main technical approach, "
            "call out tradeoffs, and discuss production constraints."
        )

    print("\nYour answer:")
    lines = []
    while True:
        line = input()
        if not line.strip() and lines:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main() -> None:
    args = _parse_args()
    db_path = args.db_path
    if args.resume_session:
        state = load_session(args.resume_session, db_path=db_path)
        state.flags["mock_llm"] = args.mock_llm
        print(f"Resuming session {state.session_id}.\n")
    elif args.resume_latest:
        state = load_latest_session(user_id=args.user_id, db_path=db_path, status=SessionStatus.ACTIVE.value)
        state.flags["mock_llm"] = args.mock_llm
        print(f"Resuming latest active session {state.session_id}.\n")
    else:
        if not args.mock_llm and not has_valid_groq_key():
            raise SystemExit(
                "Missing valid GROQ_API_KEY in .env. Add a real key, then rerun with --no-mock-llm."
            )
        state = InterviewState(
            role=RoleCode(args.role),
            user_id=args.user_id,
            target_job_id=args.target_job_id,
            max_rounds=args.rounds,
            difficulty=Difficulty(args.difficulty),
            topic=Topic(args.topic) if args.topic else None,
            flags={"mock_llm": args.mock_llm, "fixed_topic": bool(args.topic)},
        )
        mode_label = "mock" if args.mock_llm else "LLM-backed"
        print(f"Starting {args.rounds}-round {mode_label} interview for {args.role}.\n")
    save_session(state, db_path=db_path)

    while state.session_status != SessionStatus.COMPLETED:
        state = start_round(state)
        save_session(state, db_path=db_path)
        if state.session_status == SessionStatus.COMPLETED:
            break

        round_number = state.current_round + 1
        print(f"Round {round_number}/{state.max_rounds}")
        print(f"Topic: {state.topic.value if state.topic else 'general'}")
        print(f"Difficulty: {state.difficulty.value}")
        print(f"Question: {state.question}")

        answer = _read_answer(args.auto_answer, round_number)
        if not answer:
            print("No answer provided; ending session.")
            save_session(state, db_path=db_path)
            break

        state = submit_answer(state, answer)
        save_session(state, db_path=db_path)
        print(f"Score: {state.scores.total}/10")
        print(f"Feedback: {state.scores.feedback}\n")

    print("Interview complete.")
    print(f"Session ID: {state.session_id}")
    print(f"Rounds completed: {len(state.history)}")
    print(f"Skill signals: {state.skill_signals}")
    if state.flags.get("retrieval_error"):
        print(f"Retrieval error: {state.flags['retrieval_error']}")
    if state.flags.get("generation_error"):
        print(f"Generation fallback reason: {state.flags['generation_error']}")
    if state.flags.get("evaluation_error"):
        print(f"Evaluation fallback reason: {state.flags['evaluation_error']}")

    target_job = None
    if state.target_job_id:
        target_job = find_job_by_id(state.target_job_id, dataset=args.dataset)
    if target_job is None:
        target_job = find_matching_job(state.role, dataset=args.dataset)

    if target_job is not None and state.history:
        report = analyze_state_against_job(state, target_job)
        roadmap = build_learning_roadmap(report)
        state.skill_gap_report = report.model_dump()
        state.learning_roadmap = roadmap.model_dump()
        state.coach_feedback = roadmap.summary
        save_session(state, db_path=db_path)
        print("\nSkill Gap Summary:")
        print(f"- missing: {len(report.missing)}")
        print(f"- weak: {len(report.weak)}")
        print(f"- developing: {len(report.developing)}")
        print(f"- strong: {len(report.strong)}")
        print("\nCoaching Roadmap:")
        print(roadmap.summary)
        for item in roadmap.items:
            print(f"* {item.priority}. {item.skill}: {item.goal}")
            print(f"  Why: {item.why_this_matters}")
            if item.evidence_summary:
                print(f"  Evidence: {item.evidence_summary[0]}")
            if item.resources:
                top_resource = item.resources[0]
                print(f"  Start with: {top_resource.title} ({top_resource.doc_id})")


if __name__ == "__main__":
    main()
