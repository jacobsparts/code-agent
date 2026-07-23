from datetime import datetime, timedelta, timezone
import pytest

from code_agent.session_store import SessionStore


def test_fork_session_copies_events_and_preview_blobs_without_lock(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    source = store.create_session("/repo", "model-a")
    store.append_event(source, 1, "message_added", {"message": {"role": "user", "_user_content": "hello"}})
    store.append_event(source, 2, "display", {"kind": "status", "text": "ok\n"})
    store.save_preview_blob(source, "abc", "blob content")
    ok, _ = store.acquire_session_lock(source, "owner-a")
    assert ok

    forked = store.fork_session(source, cwd="/other", model="model-b")

    forked_session = store.get_session(forked)
    assert forked_session["cwd"] == "/other"
    assert forked_session["model"] == "model-b"
    assert forked_session["last_user_text"] == "hello"
    assert store.get_events(forked) == store.get_events(source)
    assert store.get_preview_blob(forked, "abc") == "blob content"
    assert store.get_session_lock(forked) is None
    assert store.get_session_lock(source)["owner"] == "owner-a"


def test_get_transcript_events_only_loads_visible_message_fields(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    store.append_event(
        session_id,
        1,
        "message_added",
        {
            "message": {
                "role": "user",
                "content": "visible",
                "_stdout": "large private output",
                "_render_segments": [{"content": "large private output"}],
            }
        },
    )
    store.append_event(session_id, 2, "display", {"kind": "status", "text": "ignored"})
    store.append_event(
        session_id,
        3,
        "message_added",
        {"message": {"role": "assistant", "content": "emit('done')", "_final_result": "done"}},
    )

    events = store.get_transcript_events(session_id)

    assert [event["seq"] for event in events] == [1, 3]
    assert events[0]["payload"]["message"] == {"role": "user", "content": "visible"}
    assert events[1]["payload"]["message"] == {"role": "assistant", "content": "emit('done')"}


def test_session_lock_blocks_other_owner_until_released(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")

    ok, lock = store.acquire_session_lock(session_id, "owner-a")
    assert ok
    assert lock is None

    ok, lock = store.acquire_session_lock(session_id, "owner-b")
    assert not ok
    assert lock["owner"] == "owner-a"

    assert store.heartbeat_session_lock(session_id, "owner-a")
    assert store.release_session_lock(session_id, "owner-a")

    ok, lock = store.acquire_session_lock(session_id, "owner-b")
    assert ok
    assert lock is None


def test_session_lock_can_be_stolen_after_expiry(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")

    ok, _ = store.acquire_session_lock(session_id, "owner-a", ttl_seconds=-1)
    assert ok
    assert store.get_session_lock(session_id) is None

    ok, lock = store.acquire_session_lock(session_id, "owner-b")
    assert ok
    assert lock is None
    assert store.get_session_lock(session_id)["owner"] == "owner-b"


def test_session_lock_same_owner_refreshes(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")

    ok, _ = store.acquire_session_lock(session_id, "owner-a", ttl_seconds=1)
    assert ok
    first = store.get_session_lock(session_id)

    ok, _ = store.acquire_session_lock(session_id, "owner-a", ttl_seconds=120)
    assert ok
    second = store.get_session_lock(session_id)

    assert second["owner"] == "owner-a"
    assert datetime.fromisoformat(second["expires_at"]) >= datetime.fromisoformat(first["expires_at"])



def test_create_session_from_messages_appends_events_and_copies_preview_blobs(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    source = store.create_session("/repo", "model-a")
    store.append_event(source, 1, "message_added", {"message": {"role": "user", "_user_content": "source"}})
    store.save_preview_blob(source, "abc", "blob content")
    messages = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": "condensed",
            "_attachments": {"file.txt": "content"},
            "_attachment_refs": {"blob": "session://preview/abc"},
            "_render_segments": [{"type": "stdout", "content": "condensed"}],
        },
        {"role": "assistant", "content": "emit('ok')"},
    ]

    created = store.create_session_from_messages(
        messages,
        cwd="/new",
        model="model-b",
        preview_blobs_from=source,
    )

    assert created != source
    assert store.get_session(created)["cwd"] == "/new"
    assert store.get_session(created)["model"] == "model-b"
    assert store.get_preview_blob(created, "abc") == "blob content"
    events = store.get_events(created)
    assert [event["seq"] for event in events] == [1, 2]
    assert [event["event_type"] for event in events] == ["message_added", "message_added"]
    first_message = events[0]["payload"]["message"]
    assert first_message["role"] == "user"
    assert first_message["_attachment_refs"] == {"blob": "session://preview/abc"}
    assert "_attachments" not in first_message
    assert store.get_events(source)[0]["payload"]["message"]["_user_content"] == "source"


def test_sessions_are_scoped_by_host(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    local = store.create_session("/repo", "model", host="local")
    remote = store.create_session("/repo", "model", host="remote.example.test")

    assert [row["session_id"] for row in store.list_sessions(cwd="/repo", host="local")] == [local]
    assert [row["session_id"] for row in store.list_sessions(cwd="/repo", host="remote.example.test")] == [remote]
    assert {row["session_id"] for row in store.list_sessions(cwd="/repo")} == {local, remote}


def test_fork_session_can_override_host(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    source = store.create_session("/repo", "model-a", host="local")

    forked = store.fork_session(source, cwd="/remote/repo", host="remote.example.test")

    forked_session = store.get_session(forked)
    assert forked_session["host"] == "remote.example.test"
    assert forked_session["cwd"] == "/remote/repo"


def test_existing_session_db_gets_local_host_column(tmp_path):
    db_path = tmp_path / "sessions.db"
    store = SessionStore(str(db_path))
    session_id = store.create_session("/repo", "model")
    with store._connect() as conn:
        conn.execute("ALTER TABLE sessions RENAME TO sessions_old")
        conn.execute("""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                cwd TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                model TEXT,
                title TEXT,
                last_user_text TEXT
            )
        """)
        conn.execute("""
            INSERT INTO sessions(session_id, cwd, created_at, updated_at, model, title, last_user_text)
            SELECT session_id, cwd, created_at, updated_at, model, title, last_user_text
            FROM sessions_old
        """)
        conn.execute("DROP TABLE sessions_old")
        conn.commit()

    migrated = SessionStore(str(db_path))

    assert migrated.get_session(session_id)["host"] == "local"


def test_preview_content_is_global_but_access_is_session_scoped(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    first = store.create_session("/repo", "model")
    second = store.create_session("/repo", "model")

    store.save_preview_blob(first, "abc", "shared")
    assert store.get_preview_blob(first, "abc") == "shared"
    assert store.get_preview_blob(second, "abc") is None

    store.save_preview_blob(second, "abc", "shared")

    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM preview_blobs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM session_preview_blobs").fetchone()[0] == 2


def test_saving_conflicting_content_for_existing_key_fails(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    first = store.create_session("/repo", "model")
    second = store.create_session("/repo", "model")
    store.save_preview_blob(first, "abc", "original")

    with pytest.raises(RuntimeError, match="Key conflict"):
        store.save_preview_blob(second, "abc", "collision")

    assert store.get_preview_blob(second, "abc") is None


def test_saving_content_overwrites_redacted_preview(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    first = store.create_session("/repo", "model")
    second = store.create_session("/repo", "model")
    store.save_preview_blob(
        first,
        "abc",
        "[Preview content redacted during database migration]",
    )

    store.save_preview_blob(second, "abc", "restored content")

    assert store.get_preview_blob(first, "abc") == "restored content"
    assert store.get_preview_blob(second, "abc") == "restored content"


def test_restored_preview_rejects_later_conflict(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    first = store.create_session("/repo", "model")
    second = store.create_session("/repo", "model")
    third = store.create_session("/repo", "model")
    store.save_preview_blob(first, "abc", "[Preview content redacted by retention policy]")
    store.save_preview_blob(second, "abc", "restored content")

    with pytest.raises(RuntimeError, match="Key conflict"):
        store.save_preview_blob(third, "abc", "different content")

    assert store.get_preview_blob(third, "abc") is None


def test_forks_and_copy_only_add_preview_associations(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    source = store.create_session("/repo", "model")
    store.save_preview_blob(source, "abc", "shared")

    forked = store.fork_session(source)
    target = store.create_session("/repo", "model")
    store.copy_preview_blobs(source, target)
    store.copy_preview_blobs(source, target)

    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM preview_blobs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM session_preview_blobs").fetchone()[0] == 3
    assert store.get_preview_blob(forked, "abc") == "shared"
    assert store.get_preview_blob(target, "abc") == "shared"
