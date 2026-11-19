"""Pruebas de las rutas de la API.

Las rutas de base de datos usan el Postgres real dentro de la transacción que se
revierte. La ruta de mensajes sustituye el grafo por una función falsa: aquí se
comprueba la ruta, no el modelo (eso está en test_memory.py).
"""
import pytest
from fastapi.testclient import TestClient

import routes.routes as r
from db.postgress import get_conn
from main import app


@pytest.fixture
def client():
    # Sin `with`: no se ejecuta lifespan, así que no hace falta la base para crear el cliente.
    return TestClient(app)


@pytest.fixture
def db_client(client, conn):
    """Cliente cuyas rutas usan la conexión de prueba, que se revierte al final."""
    app.dependency_overrides[get_conn] = lambda: conn
    yield client
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")

    assert (response.status_code, response.json()) == (200, {"status": "ok"})


# --- Chats e historial ---
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


# --- Mensajes ---
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
    assert calls == [("Who was Bolívar?", "history", "api-chat")]      # el chat_id es la conversación


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
