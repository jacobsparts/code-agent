#!/usr/bin/env python3
import json
import sqlite3
from pathlib import Path

def content_blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if (
        isinstance(content, list)
        and content
        and all(
            isinstance(block, dict) and isinstance(block.get("type"), str)
            for block in content
        )
    ):
        return content
    return [{"type": "text", "text": json.dumps(content)}]


SOURCE_VERSION = 2
MIGRATING_VERSION = 25  # SQLite user_version only supports integers.
TARGET_VERSION = 3
DATABASE = Path.home() / ".code-agent" / "sessions.db"


def migrate_message_database() -> tuple[int, int]:
    database = DATABASE
    if not database.is_file():
        raise FileNotFoundError("Database not found")

    conn = sqlite3.connect(str(database), timeout=1)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA locking_mode = EXCLUSIVE")
        conn.execute("BEGIN EXCLUSIVE")

        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version not in (SOURCE_VERSION, MIGRATING_VERSION):
            raise RuntimeError(
                f"Expected schema version {SOURCE_VERSION} or "
                f"in-progress version {MIGRATING_VERSION}, found {version}"
            )
        if version == SOURCE_VERSION:
            conn.execute(f"PRAGMA user_version = {MIGRATING_VERSION}")

        session_ids = [
            row["session_id"]
            for row in conn.execute(
                "SELECT session_id FROM sessions ORDER BY session_id"
            )
        ]
        conn.commit()

        processed = 0
        updated = 0
        for session_id in session_ids:
            conn.execute("BEGIN EXCLUSIVE")
            rows = conn.execute(
                """
                SELECT id, seq, payload_json
                FROM session_events
                WHERE session_id = ? AND event_type = 'message_added'
                ORDER BY seq
                """,
                (session_id,),
            )
            for row in rows:
                try:
                    payload = json.loads(row["payload_json"])
                    message = payload["message"]
                    if not isinstance(payload, dict) or not isinstance(message, dict):
                        raise TypeError
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise RuntimeError(
                        f"Invalid message event in session {session_id} "
                        f"at sequence {row['seq']}"
                    ) from exc

                content = message.get("content")
                blocks = content_blocks(content)
                if blocks != content:
                    message["content"] = blocks
                    conn.execute(
                        "UPDATE session_events SET payload_json = ? WHERE id = ?",
                        (json.dumps(payload, ensure_ascii=False), row["id"]),
                    )
                    updated += 1
                processed += 1
            conn.commit()

        conn.execute("BEGIN EXCLUSIVE")
        conn.execute(f"PRAGMA user_version = {TARGET_VERSION}")
        conn.commit()
        return processed, updated
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    processed, updated = migrate_message_database()
    print(
        f"Migration complete: {processed} message events processed, "
        f"{updated} updated, schema version {TARGET_VERSION}."
    )


if __name__ == "__main__":
    main()
