from urllib.parse import quote_plus
from config.settings import settings
from contextlib import asynccontextmanager
from psycopg.rows import DictRow, dict_row
import psycopg
from  fastapi import FastAPI, HTTPException
from models.models_api import CreateChat, OutputChat

DATABASE_URL = (
    f"postgresql://{settings.POSTGRES_USER}:"
    f"{quote_plus(settings.POSTGRES_PASSWORD.get_secret_value())}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB_NAME}"
)

def get_conn():
    with psycopg.Connection[DictRow].connect(DATABASE_URL, row_factory=dict_row) as conn:
        yield conn


@asynccontextmanager
async def lifespan(app: FastAPI):
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXIST chat_history (
                id SERIAL PRIMARY KEY,
                id_chat text NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
    yield

def get_history_chat(chat_id:int,conn = get_conn):
    row = conn.execute("SELECT * FROM WHERE id = %s", (chat_id)).fetchone()
    if row is None:
        raise HTTPException(404, "Chat not Found")
    return row

def create_chat(data: CreateChat, conn=get_conn):
    if data is None:
        raise HTTPException(404, "The Data is empty")

    data = conn.execute(
        "INSERT INTO chat_history (id_chat, role, content)"
        "VALUES (%(id_chat)s, %(role)s, %(content)s) RETURNING *",
        data.model_dump(),
    ).fetchone()

    return data

