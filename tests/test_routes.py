"""Tests for the API routes.

The database routes use the real Postgres inside the transaction that is rolled
back. The messages route replaces the graph with a stub: what is checked here is
the route, not the model (that is covered in test_memory.py).
"""
import pytest
from fastapi.testclient import TestClient

import routes.routes as r
from db.postgress import get_conn
from main import app


@pytest.fixture
def client():
    # No `with`: lifespan does not run, so the database is not needed to create the client.
    return TestClient(app)


@pytest.fixture
def db_client(client, conn):
    """Client whose routes use the test connection, which is rolled back at the end."""
    app.dependency_overrides[get_conn] = lambda: conn
    yield client
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")

    assert (response.status_code, response.json()) == (200, {"status": "ok"})


# --- Chats and history ---
@pytest.mark.db
def test_create_chat(db_client):
    response = db_client.post("/chats", json={"chat_id": "api-chat", "name": "My chat"})

    assert response.status_code == 201
    assert response.json() == {"chat_id": "api-chat", "name": "My chat", "summary": None}


@pytest.mark.db
def test_create_chat_twice_is_409(db_client):
    db_client.post("/chats", json={"chat_id": "api-chat", "name": "My chat"})

    response = db_client.post("/chats", json={"chat_id": "api-chat", "name": "Other"})

    assert response.status_code == 409


def test_create_chat_without_name_is_422(client):
    response = client.post("/chats", json={"chat_id": "api-chat"})

    assert response.status_code == 422


@pytest.mark.db
def test_history_of_new_chat_is_empty(db_client):
    db_client.post("/chats", json={"chat_id": "api-chat", "name": "My chat"})

    response = db_client.get("/chats/api-chat/history")

    assert (response.status_code, response.json()) == (200, [])


@pytest.mark.db
def test_history_of_unknown_chat_is_404(db_client):
    response = db_client.get("/chats/no-such-chat/history")

    assert response.status_code == 404


# --- Messages ---
QUESTION = {"user_name": "Ana", "prompt_user": "Who was Bolívar?", "topic": "history"}


def test_send_message_returns_the_answer(client, monkeypatch):
    calls = []

    def fake_answer(data, thread_id):
        calls.append((data.prompt_user, data.topic, thread_id))
        return "the answer"

    monkeypatch.setattr(r, "answer_question", fake_answer)

    response = client.post("/chats/api-chat/messages", json=QUESTION)

    assert response.status_code == 200
    assert response.json() == {"chat_id": "api-chat", "response_message": "the answer"}
    assert calls == [("Who was Bolívar?", "history", "api-chat")]      # the chat_id is the conversation


def test_send_message_with_invalid_topic_is_400(client, monkeypatch):
    def fake_answer(data, thread_id):
        raise ValueError("Invalid topic selected.")

    monkeypatch.setattr(r, "answer_question", fake_answer)

    response = client.post("/chats/api-chat/messages", json={**QUESTION, "topic": "cooking"})

    assert response.status_code == 400
    assert "Invalid topic" in response.json()["detail"]


def test_send_message_without_question_is_422(client):
    response = client.post("/chats/api-chat/messages", json={"user_name": "Ana", "topic": "history"})

    assert response.status_code == 422


def test_send_message_when_database_is_down_is_503(client, monkeypatch):
    import psycopg

    def fake_answer(data, thread_id):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(r, "answer_question", fake_answer)

    response = client.post("/chats/api-chat/messages", json=QUESTION)

    assert response.status_code == 503
