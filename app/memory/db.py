"""Phase 10 SQLite persistence for interview sessions."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import MEMORY_DB_PATH
from app.schemas.interview_state import InterviewState, RoundRecord


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_db_path(db_path: str | Path | None = MEMORY_DB_PATH) -> Path:
    return Path(db_path or MEMORY_DB_PATH)


def connect(db_path: str | Path | None = MEMORY_DB_PATH) -> sqlite3.Connection:
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path | None = MEMORY_DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                role TEXT NOT NULL,
                target_job_id TEXT,
                session_status TEXT NOT NULL,
                current_round INTEGER NOT NULL,
                max_rounds INTEGER NOT NULL,
                round_type TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                topic TEXT,
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rounds (
                session_id TEXT NOT NULL,
                round_number INTEGER NOT NULL,
                round_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (session_id, round_number),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS skill_snapshots (
                session_id TEXT PRIMARY KEY,
                skill_signals_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            """
        )


def _session_payload(state: InterviewState) -> tuple[Any, ...]:
    return (
        state.session_id,
        state.user_id,
        state.role.value,
        state.target_job_id,
        state.session_status.value,
        state.current_round,
        state.max_rounds,
        state.round_type.value,
        state.difficulty.value,
        state.topic.value if state.topic else None,
        state.model_dump_json(),
    )


def save_session(state: InterviewState, db_path: str | Path | None = MEMORY_DB_PATH) -> None:
    init_db(db_path)
    now = _utc_now()
    with connect(db_path) as conn:
        existing = conn.execute(
            "SELECT created_at FROM sessions WHERE session_id = ?",
            (state.session_id,),
        ).fetchone()
        created_at = existing["created_at"] if existing else now
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions (
                session_id, user_id, role, target_job_id, session_status,
                current_round, max_rounds, round_type, difficulty, topic,
                state_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _session_payload(state) + (created_at, now),
        )
        for record in state.history:
            save_round(record, state.session_id, conn=conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO skill_snapshots (
                session_id, skill_signals_json, created_at
            ) VALUES (?, ?, ?)
            """,
            (state.session_id, json.dumps(state.skill_signals), now),
        )


def save_round(
    record: RoundRecord,
    session_id: str,
    db_path: str | Path | None = MEMORY_DB_PATH,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    def _write(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            INSERT OR REPLACE INTO rounds (
                session_id, round_number, round_json, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (session_id, record.round, record.model_dump_json(), _utc_now()),
        )

    if conn is not None:
        _write(conn)
        return

    init_db(db_path)
    with connect(db_path) as connection:
        _write(connection)


def load_session(session_id: str, db_path: str | Path | None = MEMORY_DB_PATH) -> InterviewState:
    init_db(db_path)
    with connect(db_path) as conn:
        session_row = conn.execute(
            "SELECT state_json FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        round_rows = conn.execute(
            "SELECT round_json FROM rounds WHERE session_id = ? ORDER BY round_number",
            (session_id,),
        ).fetchall()
        skill_row = conn.execute(
            "SELECT skill_signals_json FROM skill_snapshots WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if session_row is None:
        raise KeyError(f"Session not found: {session_id}")
    state = InterviewState.model_validate_json(session_row["state_json"])
    if round_rows:
        state.history = [RoundRecord.model_validate_json(row["round_json"]) for row in round_rows]
    if skill_row is not None:
        state.skill_signals = json.loads(skill_row["skill_signals_json"])
    return state


def get_session_record(session_id: str, db_path: str | Path | None = MEMORY_DB_PATH) -> dict[str, Any]:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT session_id, user_id, role, target_job_id, session_status,
                   current_round, max_rounds, round_type, difficulty, topic,
                   created_at, updated_at
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
    if row is None:
        raise KeyError(f"Session not found: {session_id}")
    return dict(row)


def list_sessions(
    user_id: Optional[str] = None,
    db_path: str | Path | None = MEMORY_DB_PATH,
) -> list[dict]:
    init_db(db_path)
    query = (
        "SELECT session_id, user_id, role, target_job_id, session_status, "
        "current_round, max_rounds, round_type, difficulty, topic, created_at, updated_at "
        "FROM sessions"
    )
    params: tuple[str, ...] = ()
    if user_id:
        query += " WHERE user_id = ?"
        params = (user_id,)
    query += " ORDER BY updated_at DESC"

    with connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def load_latest_session(
    user_id: Optional[str] = None,
    db_path: str | Path | None = MEMORY_DB_PATH,
    status: Optional[str] = None,
) -> InterviewState:
    init_db(db_path)
    clauses = []
    params: list[str] = []
    if user_id:
        clauses.append("user_id = ?")
        params.append(user_id)
    if status:
        clauses.append("session_status = ?")
        params.append(status)

    query = "SELECT session_id FROM sessions"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY updated_at DESC LIMIT 1"

    with connect(db_path) as conn:
        row = conn.execute(query, tuple(params)).fetchone()
    if row is None:
        scope = f"user_id={user_id!r}" if user_id else "all users"
        raise KeyError(f"No saved sessions found for {scope}")
    return load_session(str(row["session_id"]), db_path=db_path)


def delete_session(session_id: str, db_path: str | Path | None = MEMORY_DB_PATH) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("DELETE FROM skill_snapshots WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM rounds WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
