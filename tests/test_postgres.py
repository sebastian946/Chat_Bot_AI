"""Tests for the database functions.

They use the real Postgres (the one from docker compose). Each test runs inside a
transaction that is rolled back at the end, so they leave no data or tables behind.
If the database is not reachable, the tests are skipped instead of failing.
"""
import pytest
from fastapi import HTTPException

import db.postgress as p
from models.models_api import CreateChat, PushChatHistory

pytestmark = pytest.mark.db

CHAT = "test-chat"


@pytest.fixture
def chat(conn):
    """A chat with five messages, m0..m4, with ids id-0..id-4."""
    p.create_chat(CreateChat(chat_id=CHAT, name="Test"), conn)
    for i, role in enumerate(["user", "assistant", "user", "assistant", "user"]):
        add_message(conn, CHAT, role, f"m{i}", f"id-{i}")
    return CHAT


def add_message(conn, chat_id, role, content, message_id=None):
    data = PushChatHistory(chat_id=chat_id, role=role, content=content, message_id=message_id)
    return p.push_chat_history(data, conn)


def get_summary(conn, chat_id):
    return conn.execute(
        "SELECT summary FROM existing_chats WHERE chat_id = %s", (chat_id,)
    ).fetchone()["summary"]


# --- Tables ---
def test_create_tables_can_run_twice(conn):
    p.create_tables(conn)      # the fixture already ran it once

    assert p.get_history_chat  # getting here without an error is the check


# --- Chats ---
def test_create_chat_returns_the_row_without_summary(conn):
    row = p.create_chat(CreateChat(chat_id=CHAT, name="Test"), conn)

    assert row == {"chat_id": CHAT, "name": "Test", "summary": None}


def test_ensure_chat_creates_once_and_keeps_the_first_name(conn):
    p.ensure_chat(CHAT, "First", conn)
    p.ensure_chat(CHAT, "Second", conn)

    rows = conn.execute("SELECT name FROM existing_chats WHERE chat_id = %s", (CHAT,)).fetchall()
    assert rows == [{"name": "First"}]


def test_create_chat_twice_is_409(conn):
    p.create_chat(CreateChat(chat_id=CHAT, name="Test"), conn)

    with pytest.raises(HTTPException) as error:
        p.create_chat(CreateChat(chat_id=CHAT, name="Other"), conn)
    assert error.value.status_code == 409


# --- Messages ---
def test_push_stores_the_message_as_not_summarized(conn):
    p.create_chat(CreateChat(chat_id=CHAT, name="Test"), conn)

    row = add_message(conn, CHAT, "user", "hello", "id-x")

    assert (row["id_chat"], row["role"], row["content"], row["message_id"]) == (CHAT, "user", "hello", "id-x")
    assert row["summarized"] is False
    assert row["created_at"] is not None


def test_push_to_unknown_chat_is_404(conn):
    with pytest.raises(HTTPException) as error:
        add_message(conn, "no-such-chat", "user", "hello")
    assert error.value.status_code == 404


def test_push_repeated_message_id_is_409(chat, conn):
    with pytest.raises(HTTPException) as error:
        add_message(conn, chat, "user", "again", "id-0")
    assert error.value.status_code == 409


def test_history_is_returned_oldest_first(chat, conn):
    history = p.get_history_chat(chat, conn)

    assert [row["content"] for row in history] == ["m0", "m1", "m2", "m3", "m4"]


def test_history_of_empty_chat_is_empty_list(conn):
    p.create_chat(CreateChat(chat_id=CHAT, name="Test"), conn)

    assert p.get_history_chat(CHAT, conn) == []


def test_history_of_unknown_chat_is_404(conn):
    with pytest.raises(HTTPException) as error:
        p.get_history_chat("no-such-chat", conn)
    assert error.value.status_code == 404


# --- Summary ---
def test_summary_is_saved_and_only_listed_messages_are_marked(chat, conn):
    marked = p.update_chat_history_with_summary(chat, "the summary", ["id-0", "id-1", "id-2"], conn)

    assert marked == 3
    assert get_summary(conn, chat) == "the summary"
    assert [(r["content"], r["summarized"]) for r in p.get_history_chat(chat, conn)] == [
        ("m0", True), ("m1", True), ("m2", True), ("m3", False), ("m4", False),
    ]


def test_pending_history_excludes_summarized_messages(chat, conn):
    p.update_chat_history_with_summary(chat, "the summary", ["id-0", "id-1", "id-2"], conn)

    pending = p.get_history_chat(chat, conn, include_summarized=False)

    assert [row["content"] for row in pending] == ["m3", "m4"]


def test_unknown_message_ids_are_ignored(chat, conn):
    marked = p.update_chat_history_with_summary(chat, "the summary", ["id-0", "does-not-exist"], conn)

    assert marked == 1


def test_second_summary_replaces_the_first_and_marks_only_new_messages(chat, conn):
    p.update_chat_history_with_summary(chat, "summary v1", ["id-0", "id-1"], conn)

    marked = p.update_chat_history_with_summary(chat, "summary v2", ["id-0", "id-1", "id-2", "id-3"], conn)

    assert marked == 2
    assert get_summary(conn, chat) == "summary v2"


def test_empty_id_list_saves_summary_and_marks_nothing(chat, conn):
    marked = p.update_chat_history_with_summary(chat, "the summary", [], conn)

    assert marked == 0
    assert get_summary(conn, chat) == "the summary"


def test_summary_does_not_touch_messages_of_another_chat(chat, conn):
    p.create_chat(CreateChat(chat_id="other-chat", name="Other"), conn)
    add_message(conn, "other-chat", "user", "other message", "other-id")

    marked = p.update_chat_history_with_summary(chat, "the summary", ["id-0", "other-id"], conn)

    assert marked == 1
    assert p.get_history_chat("other-chat", conn)[0]["summarized"] is False
    assert get_summary(conn, "other-chat") is None


# --- Memory for the model ---
def test_memory_of_unknown_chat_is_none(conn):
    assert p.get_chat_memory("no-such-chat", conn) is None


def test_memory_without_summary_has_every_message_pending(chat, conn):
    memory = p.get_chat_memory(chat, conn)

    assert memory["summary"] is None
    assert memory["summarized_message_ids"] == []
    assert memory["last_summarized_message_id"] is None
    assert [row["content"] for row in memory["pending"]] == ["m0", "m1", "m2", "m3", "m4"]


def test_memory_with_summary_has_only_unsummarized_messages_pending(chat, conn):
    p.update_chat_history_with_summary(chat, "the summary", ["id-0", "id-1", "id-2"], conn)

    memory = p.get_chat_memory(chat, conn)

    assert memory["summary"] == "the summary"
    assert memory["summarized_message_ids"] == ["id-0", "id-1", "id-2"]
    assert memory["last_summarized_message_id"] == "id-2"
    assert memory["pending"] == [
        {"message_id": "id-3", "role": "assistant", "content": "m3"},
        {"message_id": "id-4", "role": "user", "content": "m4"},
    ]


def test_summary_for_unknown_chat_is_404(conn):
    with pytest.raises(HTTPException) as error:
        p.update_chat_history_with_summary("no-such-chat", "the summary", ["id-0"], conn)
    assert error.value.status_code == 404
