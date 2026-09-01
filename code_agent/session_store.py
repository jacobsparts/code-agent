import copy
import json
import os
import socket
import sqlite3
import uuid

from .persisted_preview_state import PersistedPreviewTransitions
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCHEMA_VERSION = 3


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_db_path() -> Path:
    if override := os.getenv("CODE_AGENT_SESSION_DB"):
        path = Path(override).expanduser()
    else:
        path = Path.home() / ".code-agent" / "sessions.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class SessionStore:
    def __init__(self, db_path: str | None = None):
        self.db_path = str(Path(db_path).expanduser()) if db_path else str(resolve_db_path())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version == 2:
                raise RuntimeError(
                    "Database migration required; run "
                    "./code_agent/utility/db_migrate_to_v3.py"
                )
            if version == 25:
                raise RuntimeError("Database migration in progress")
            if version not in (0, SCHEMA_VERSION):
                raise RuntimeError("Unsupported schema")
            preview_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(preview_blobs)").fetchall()
            }
            if "session_id" in preview_columns:
                raise RuntimeError("Migration required")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    host TEXT NOT NULL DEFAULT 'local',
                    cwd TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    model TEXT,
                    title TEXT,
                    last_user_text TEXT
                )
            """)
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "host" not in columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN host TEXT NOT NULL DEFAULT 'local'")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE(session_id, seq),
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS preview_blobs (
                    key TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    content TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_preview_blobs (
                    session_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(session_id, key),
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id),
                    FOREIGN KEY(key) REFERENCES preview_blobs(key)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_locks (
                    session_id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    pid INTEGER NOT NULL,
                    hostname TEXT,
                    acquired_at TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_host_cwd ON sessions(host, cwd)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_events_session_seq ON session_events(session_id, seq)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_locks_expires_at ON session_locks(expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_preview_blobs_key ON session_preview_blobs(key)")
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
            conn.execute("PRAGMA journal_mode = WAL")

    def create_session(self, cwd: str, model: str | None = None, host: str = "local") -> str:
        session_id = str(uuid.uuid4())
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id, host, cwd, created_at, updated_at, model)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, host, cwd, now, now, model),
            )
            conn.commit()
        return session_id

    def fork_session(self, source_session_id: str, *, cwd: str | None = None, model: str | None = None, host: str | None = None) -> str:
        new_session_id = str(uuid.uuid4())
        now = utc_now_iso()
        with self._connect() as conn:
            source = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (source_session_id,),
            ).fetchone()
            if source is None:
                raise ValueError("Session not found")

            conn.execute(
                """
                INSERT INTO sessions(session_id, host, cwd, created_at, updated_at, model, title, last_user_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_session_id,
                    host if host is not None else source["host"],
                    cwd if cwd is not None else source["cwd"],
                    now,
                    now,
                    model if model is not None else source["model"],
                    source["title"],
                    source["last_user_text"],
                ),
            )
            conn.execute(
                """
                INSERT INTO session_events(session_id, seq, created_at, event_type, payload_json)
                SELECT ?, seq, created_at, event_type, payload_json
                FROM session_events
                WHERE session_id = ?
                ORDER BY seq ASC
                """,
                (new_session_id, source_session_id),
            )
            conn.execute(
                """
                INSERT INTO session_preview_blobs(session_id, key, created_at)
                SELECT ?, key, created_at
                FROM session_preview_blobs
                WHERE session_id = ?
                """,
                (new_session_id, source_session_id),
            )
            conn.commit()
        return new_session_id
    def copy_preview_blobs(self, source_session_id: str, target_session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO session_preview_blobs(session_id, key, created_at)
                SELECT ?, key, created_at
                FROM session_preview_blobs
                WHERE session_id = ?
                """,
                (target_session_id, source_session_id),
            )
            conn.commit()

    @staticmethod
    def _message_payload_for_event(msg: dict) -> dict:
        return {
            key: value
            for key, value in msg.items()
            if key != "_attachments"
        }

    def create_session_from_messages(
        self,
        messages: list[dict],
        *,
        cwd: str,
        model: str | None = None,
        host: str = "local",
        preview_blobs_from: str | None = None,
    ) -> str:
        session_id = self.create_session(cwd, model, host)
        if preview_blobs_from:
            self.copy_preview_blobs(preview_blobs_from, session_id)
        seq = 1
        for msg in messages[1:]:
            self.append_event(session_id, seq, "message_added", {"message": self._message_payload_for_event(msg)})
            seq += 1
        return session_id


    def _lock_expiry(self, ttl_seconds: int) -> str:
        return (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()

    @staticmethod
    def default_lock_owner() -> str:
        return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4()}"

    def acquire_session_lock(self, session_id: str, owner: str, ttl_seconds: int = 3600) -> tuple[bool, dict | None]:
        now = utc_now_iso()
        expires_at = self._lock_expiry(ttl_seconds)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    "SELECT * FROM session_locks WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if existing is not None and existing["owner"] != owner:
                    parsed_expiry = datetime.fromisoformat(existing["expires_at"])
                    if parsed_expiry.tzinfo is None:
                        parsed_expiry = parsed_expiry.replace(tzinfo=timezone.utc)
                    if parsed_expiry > datetime.now(timezone.utc):
                        conn.commit()
                        return False, dict(existing)

                conn.execute(
                    """
                    INSERT INTO session_locks(session_id, owner, pid, hostname, acquired_at, heartbeat_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        owner = excluded.owner,
                        pid = excluded.pid,
                        hostname = excluded.hostname,
                        acquired_at = excluded.acquired_at,
                        heartbeat_at = excluded.heartbeat_at,
                        expires_at = excluded.expires_at
                    """,
                    (session_id, owner, os.getpid(), socket.gethostname(), now, now, expires_at),
                )
                conn.commit()
                return True, None
            except Exception:
                conn.rollback()
                raise

    def heartbeat_session_lock(self, session_id: str, owner: str, ttl_seconds: int = 3600) -> bool:
        now = utc_now_iso()
        expires_at = self._lock_expiry(ttl_seconds)
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE session_locks
                SET heartbeat_at = ?, expires_at = ?
                WHERE session_id = ? AND owner = ?
                """,
                (now, expires_at, session_id, owner),
            )
            conn.commit()
            return cur.rowcount > 0

    def release_session_lock(self, session_id: str, owner: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM session_locks WHERE session_id = ? AND owner = ?",
                (session_id, owner),
            )
            conn.commit()
            return cur.rowcount > 0

    def get_session_lock(self, session_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM session_locks WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        lock = dict(row)
        parsed_expiry = datetime.fromisoformat(lock["expires_at"])
        if parsed_expiry.tzinfo is None:
            parsed_expiry = parsed_expiry.replace(tzinfo=timezone.utc)
        if parsed_expiry <= datetime.now(timezone.utc):
            return None
        return lock

    def append_event(self, session_id: str, seq: int, event_type: str, payload: dict) -> int:
        now = utc_now_iso()
        payload_json = json.dumps(payload, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_events(session_id, seq, created_at, event_type, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, seq, now, event_type, payload_json),
            )
            last_user_text = None
            if event_type == "message_added":
                msg = payload.get("message", {})
                if msg.get("role") == "user" and msg.get("_user_content"):
                    last_user_text = msg["_user_content"]
            conn.execute(
                """
                UPDATE sessions
                SET updated_at = ?, last_user_text = COALESCE(?, last_user_text)
                WHERE session_id = ?
                """,
                (now, last_user_text, session_id),
            )
            conn.commit()
        return seq

    @staticmethod
    def _active_persisted_preview_state(event_rows, associated_preview_keys) -> tuple[dict, dict, int]:
        transitions = PersistedPreviewTransitions()
        projected_ranges = []
        projected_snapshots = {0: []}

        def apply_placement(_preview_event_seq, _definition, source_start_seq, source_end_seq):
            overlapping = [
                (index, start, end)
                for index, (start, end) in enumerate(projected_ranges)
                if not (end < source_start_seq or start > source_end_seq)
            ]
            if not overlapping:
                return False
            indexes = [item[0] for item in overlapping]
            if indexes != list(range(indexes[0], indexes[-1] + 1)):
                return False
            if overlapping[0][1] != source_start_seq or overlapping[-1][2] != source_end_seq:
                return False
            if any(start < source_start_seq or end > source_end_seq for _, start, end in overlapping):
                return False
            for previous, current in zip(overlapping, overlapping[1:]):
                if current[1] <= previous[2]:
                    return False
            projected_ranges[indexes[0]:indexes[-1] + 1] = [(source_start_seq, source_end_seq)]
            return True

        for row in event_rows:
            seq = row["seq"]
            event_type = row["event_type"]
            try:
                payload = json.loads(row["payload_json"])
            except Exception:
                payload = None

            if event_type == "message_added" and isinstance(payload, dict):
                if isinstance(payload.get("message"), dict):
                    projected_ranges.append((seq, seq))
            elif event_type == "rewind" and isinstance(payload, dict):
                target_seq = payload.get("target_seq")
                if type(target_seq) is int:
                    projected_ranges = copy.deepcopy(
                        projected_snapshots.get(target_seq, projected_snapshots[0])
                    )
            elif event_type == "exec":
                projected_ranges = []

            transitions.apply(
                seq=seq,
                event_type=event_type,
                payload=payload,
                has_preview_blob=lambda key: key in associated_preview_keys,
                apply_placement=apply_placement,
            )
            projected_snapshots[seq] = copy.deepcopy(projected_ranges)

        state = transitions.state
        return state.definitions, state.active_placements, state.exec_start_seq

    def append_preview_events(
        self,
        session_id: str,
        *,
        preview_key: str,
        summary: str,
        source_start_seq: int,
        source_end_seq: int,
        expected_next_seq: int | None = None,
        expected_exec_start_seq: int,
        expected_definitions: dict,
        expected_active_placements: dict,
        preview_content: str | None = None,
    ) -> tuple[int, int]:
        if expected_next_seq is not None and type(expected_next_seq) is not int:
            raise ValueError("expected next sequence must be an integer")
        if type(expected_exec_start_seq) is not int:
            raise ValueError("expected exec sequence must be an integer")
        if not isinstance(expected_definitions, dict):
            raise ValueError("expected preview definitions must be a dictionary")
        if not isinstance(expected_active_placements, dict):
            raise ValueError("expected active placements must be a dictionary")
        if not isinstance(preview_key, str) or not preview_key:
            raise ValueError("preview key must be a non-empty string")
        if preview_content is not None and not isinstance(preview_content, str):
            raise ValueError("preview content must be a string")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("preview summary must be a non-empty string")
        if type(source_start_seq) is not int or type(source_end_seq) is not int:
            raise ValueError("preview source boundaries must be integers")
        if source_start_seq > source_end_seq:
            raise ValueError("invalid preview source range")

        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM session_events WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                created_seq = int(row["max_seq"]) + 1
                placed_seq = created_seq + 1
                if expected_next_seq is not None and expected_next_seq != created_seq:
                    raise ValueError("stale preview event sequence")

                event_rows = conn.execute(
                    """
                    SELECT seq, event_type, payload_json
                    FROM session_events
                    WHERE session_id = ?
                    ORDER BY seq
                    """,
                    (session_id,),
                ).fetchall()
                associated_preview_keys = {
                    item["key"]
                    for item in conn.execute(
                        "SELECT key FROM session_preview_blobs WHERE session_id = ?",
                        (session_id,),
                    ).fetchall()
                }
                active_definitions, active_placements, active_exec_start_seq = (
                    self._active_persisted_preview_state(
                        event_rows,
                        associated_preview_keys,
                    )
                )
                if (
                    active_exec_start_seq != expected_exec_start_seq
                    or active_definitions != expected_definitions
                    or active_placements != expected_active_placements
                ):
                    raise ValueError("stale persisted preview state")
                if source_end_seq >= created_seq:
                    raise ValueError("preview source range must precede preview creation")
                if source_start_seq <= active_exec_start_seq:
                    raise ValueError("preview placement crosses the active exec boundary")
                for start, end in active_placements:
                    if source_start_seq <= end and start <= source_end_seq:
                        equal = source_start_seq == start and source_end_seq == end
                        contains = source_start_seq <= start and end <= source_end_seq
                        contained = start <= source_start_seq and source_end_seq <= end
                        if contained and not equal:
                            raise ValueError("preview range is inside an active placement")
                        if not contains and not equal:
                            raise ValueError("preview range partially overlaps an active placement")

                blob = conn.execute(
                    "SELECT content FROM preview_blobs WHERE key = ?",
                    (preview_key,),
                ).fetchone()
                associated = conn.execute(
                    "SELECT 1 FROM session_preview_blobs WHERE session_id = ? AND key = ?",
                    (session_id, preview_key),
                ).fetchone()
                if preview_content is None:
                    if associated is None:
                        raise ValueError("preview blob is not associated with session")
                else:
                    if blob is None:
                        conn.execute(
                            "INSERT INTO preview_blobs(key, created_at, content) VALUES (?, ?, ?)",
                            (preview_key, now, preview_content),
                        )
                    elif blob["content"].startswith("[Preview content redacted"):
                        conn.execute(
                            "UPDATE preview_blobs SET created_at = ?, content = ? WHERE key = ?",
                            (now, preview_content, preview_key),
                        )
                    elif blob["content"] != preview_content:
                        raise RuntimeError("Key conflict")
                    if associated is None:
                        conn.execute(
                            """
                            INSERT INTO session_preview_blobs(session_id, key, created_at)
                            VALUES (?, ?, ?)
                            """,
                            (session_id, preview_key, now),
                        )

                created_payload = json.dumps(
                    {"preview_key": preview_key, "summary": summary},
                    ensure_ascii=False,
                )
                placed_payload = json.dumps(
                    {
                        "preview_event_seq": created_seq,
                        "source_start_seq": source_start_seq,
                        "source_end_seq": source_end_seq,
                    },
                    ensure_ascii=False,
                )
                conn.execute(
                    """
                    INSERT INTO session_events(session_id, seq, created_at, event_type, payload_json)
                    VALUES (?, ?, ?, 'preview_created', ?)
                    """,
                    (session_id, created_seq, now, created_payload),
                )
                conn.execute(
                    """
                    INSERT INTO session_events(session_id, seq, created_at, event_type, payload_json)
                    VALUES (?, ?, ?, 'preview_placed', ?)
                    """,
                    (session_id, placed_seq, now, placed_payload),
                )
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return created_seq, placed_seq

    def get_events(self, session_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT seq, created_at, event_type, payload_json
                FROM session_events
                WHERE session_id = ?
                ORDER BY seq ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "seq": row["seq"],
                "created_at": row["created_at"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def get_transcript_events(self, session_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    seq,
                    created_at,
                    json_extract(payload_json, '$.message.role') AS role,
                    json_extract(payload_json, '$.message.content') AS content
                FROM session_events
                WHERE session_id = ? AND event_type = 'message_added'
                ORDER BY seq ASC
                """,
                (session_id,),
            )
            return [
                {
                    "seq": row["seq"],
                    "created_at": row["created_at"],
                    "event_type": "message_added",
                    "payload": {
                        "message": {
                            "role": row["role"],
                            "content": json.loads(row["content"]),
                        }
                    },
                }
                for row in rows
            ]

    def get_session(self, session_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_next_seq(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM session_events WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["max_seq"]) + 1

    def list_sessions(self, cwd: str | None = None, limit: int = 100, host: str | None = None) -> list[dict]:
        query = "SELECT * FROM sessions"
        args = []
        clauses = []
        if host is not None:
            clauses.append("host = ?")
            args.append(host)
        if cwd is not None:
            clauses.append("cwd = ?")
            args.append(cwd)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC LIMIT ?"
        args.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, args).fetchall()
        return [dict(row) for row in rows]

    def save_preview_blob(self, session_id: str, key: str, content: str) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """
                SELECT b.content, s.session_id IS NOT NULL AS associated
                FROM preview_blobs AS b
                LEFT JOIN session_preview_blobs AS s
                    ON s.key = b.key AND s.session_id = ?
                WHERE b.key = ?
                """,
                (session_id, key),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO preview_blobs(key, created_at, content)
                    VALUES (?, ?, ?)
                    """,
                    (key, now, content),
                )
            elif existing["content"].startswith("[Preview content redacted"):
                conn.execute(
                    """
                    UPDATE preview_blobs
                    SET created_at = ?, content = ?
                    WHERE key = ?
                    """,
                    (now, content, key),
                )
            elif existing["content"] != content:
                raise RuntimeError("Key conflict")
            if existing is None or not existing["associated"]:
                conn.execute(
                    """
                    INSERT INTO session_preview_blobs(session_id, key, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (session_id, key, now),
                )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            conn.commit()

    def has_preview_blob(self, session_id: str, key: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM session_preview_blobs
                WHERE session_id = ? AND key = ?
                """,
                (session_id, key),
            ).fetchone()
        return row is not None

    def get_preview_blob(self, session_id: str, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT b.content
                FROM session_preview_blobs AS s
                JOIN preview_blobs AS b ON b.key = s.key
                WHERE s.session_id = ? AND s.key = ?
                """,
                (session_id, key),
            ).fetchone()
        return row["content"] if row else None
