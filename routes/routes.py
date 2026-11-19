import psycopg
from fastapi import APIRouter, Depends, HTTPException
from psycopg.rows import DictRow

from db.postgress import create_chat, get_conn, get_history_chat
from llm.graph import answer_question
from models.models_api import CreateChat, OutputChat
from models.models_llm import ModelInput

router = APIRouter(prefix="/chats", tags=["chats"])


@router.post("", status_code=201)
def create_chat_route(data: CreateChat, conn: psycopg.Connection[DictRow] = Depends(get_conn)):
    """Create a chat. 409 if the chat_id already exists."""
    return create_chat(data, conn)


@router.get("/{chat_id}/history")
def read_history_route(chat_id: str, conn: psycopg.Connection[DictRow] = Depends(get_conn)):
    """All the messages of a chat, oldest first. 404 if the chat does not exist."""
    return get_history_chat(chat_id, conn)


@router.post("/{chat_id}/messages", response_model=OutputChat)
def send_message_route(chat_id: str, data: ModelInput):
    """Send a question to the expert of the chosen topic and return its answer.

    The chat is created on its first message if it does not exist. The question, the
    answer and the summary are stored by the graph itself.
    """
    try:
        answer = answer_question(data, thread_id=chat_id)
    except ValueError as error:                 # invalid topic
        raise HTTPException(400, str(error))
    except psycopg.OperationalError:
        raise HTTPException(503, "Database not available")
    return OutputChat(chat_id=chat_id, response_message=answer)
