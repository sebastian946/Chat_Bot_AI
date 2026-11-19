"""Pruebas de las funciones de base de datos.

Usan el Postgres real (el del docker compose). Cada prueba corre dentro de una
transacción que se revierte al final, así que no dejan datos ni tablas.
Si la base de datos no está disponible, las pruebas se saltan en vez de fallar.
"""
import psycopg
import pytest
from fastapi import HTTPException
from psycopg.rows import DictRow, dict_row

import db.postgress as p
from models.models_api import CreateChat, PushChatHistory

pytestmark = pytest.mark.db

CHAT = "test-chat"


@pytest.fixture(scope="session")
def database_is_up():
    """Se comprueba una sola vez por ejecución, para no esperar el timeout en cada prueba."""
    try:
        psycopg.connect(p.DATABASE_URL, connect_timeout=3).close()
    except psycopg.OperationalError:
        return False
    return True


@pytest.fixture
def conn(database_is_up):
    if not database_is_up:
        pytest.skip("Postgres no está disponible (levántalo con: docker compose up -d db)")
    connection = psycopg.Connection[DictRow].connect(p.DATABASE_URL, row_factory=dict_row)
    try:
        p.create_tables(connection)
        yield connection
    finally:
        connection.rollback()      # nada de lo que hizo la prueba se guarda
        connection.close()


@pytest.fixture
def chat(conn):
    """Un chat con cinco mensajes, m0..m4, con ids id-0..id-4."""
    p.create_chat(CreateChat(chat_id=CHAT, name="Test"), conn)
    for i, role in enumerate(["user", "assistant", "user", "assistant", "user"]):
        add_message(conn, CHAT, role, f"m{i}", f"id-{i}")
    return CHAT


def add_message(conn, chat_id, role, content, message_id=None):
    data = PushChatHistory(
        chat_id=chat_id, role=role, content=content, created_at="", message_id=message_id
    )
    return p.push_chat_history(data, conn)


def get_summary(conn, chat_id):
    return conn.execute(
        "SELECT summary FROM existing_chats WHERE chat_id = %s", (chat_id,)
    ).fetchone()["summary"]


# --- Tablas ---
def test_create_tables_can_run_twice(conn):
    p.create_tables(conn)      # el fixture ya la ejecutó una vez

    assert p.get_history_chat  # llegar aquí sin error es la comprobación


# --- Chats ---
def test_create_chat_returns_the_row_without_summary(conn):
    row = p.create_chat(CreateChat(chat_id=CHAT, name="Test"), conn)

    assert row == {"chat_id": CHAT, "name": "Test", "summary": None}


def test_create_chat_twice_is_409(conn):
    p.create_chat(CreateChat(chat_id=CHAT, name="Test"), conn)

    with pytest.raises(HTTPException) as error:
        p.create_chat(CreateChat(chat_id=CHAT, name="Other"), conn)
    assert error.value.status_code == 409


# --- Mensajes ---
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


# --- Resumen ---
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


def test_summary_for_unknown_chat_is_404(conn):
    with pytest.raises(HTTPException) as error:
        p.update_chat_history_with_summary("no-such-chat", "the summary", ["id-0"], conn)
    assert error.value.status_code == 404
