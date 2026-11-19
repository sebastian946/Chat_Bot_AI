from urllib.parse import quote_plus
from config.settings import settings
from contextlib import asynccontextmanager
from psycopg.rows import DictRow, dict_row
from psycopg import errors
import psycopg
from  fastapi import FastAPI, HTTPException
from models.models_api import CreateChat, PushChatHistory

DATABASE_URL = (
    f"postgresql://{settings.POSTGRES_USER}:"
    f"{quote_plus(settings.POSTGRES_PASSWORD.get_secret_value())}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB_NAME}"
)

def get_conn():
    # Commits when the request ends well, rolls back if it raises.
    with psycopg.Connection[DictRow].connect(DATABASE_URL, row_factory=dict_row) as conn:
        yield conn


def create_tables(conn: psycopg.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS existing_chats (
            chat_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            summary TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            id_chat TEXT NOT NULL
                REFERENCES existing_chats(chat_id) ON DELETE RESTRICT,
            message_id TEXT UNIQUE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            summarized BOOLEAN NOT NULL DEFAULT false,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    # CREATE TABLE IF NOT EXISTS does not add columns to a table that already
    # exists, so the columns added later are also declared here.
    conn.execute("ALTER TABLE existing_chats ADD COLUMN IF NOT EXISTS summary TEXT")
    conn.execute("ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS message_id TEXT UNIQUE")
    conn.execute("ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS summarized BOOLEAN NOT NULL DEFAULT false")


@asynccontextmanager
async def lifespan(app: FastAPI):
    with psycopg.connect(DATABASE_URL) as conn:
        create_tables(conn)
    yield


# These functions receive the connection from the route, which gets it with
# `conn: psycopg.Connection[DictRow] = Depends(get_conn)`.

def get_history_chat(
    chat_id: str,
    conn: psycopg.Connection[DictRow],
    include_summarized: bool = True,
) -> list[DictRow]:
    """The messages of a chat, oldest first. 404 if the chat does not exist.

    With include_summarized=False only the messages the summary does not cover
    are returned: together with the chat's summary, that is the model's context.
    """
    chat = conn.execute(
        "SELECT chat_id FROM existing_chats WHERE chat_id = %s", (chat_id,)
    ).fetchone()
    if chat is None:
        raise HTTPException(404, "Chat not found")

    return conn.execute(
        "SELECT id, id_chat, message_id, role, content, summarized, created_at FROM chat_history "
        "WHERE id_chat = %s AND (%s OR NOT summarized) ORDER BY created_at, id",
        (chat_id, include_summarized),
    ).fetchall()


def create_chat(data: CreateChat, conn: psycopg.Connection[DictRow]) -> DictRow | None:
    try:
        return conn.execute(
            "INSERT INTO existing_chats (chat_id, name) "
            "VALUES (%(chat_id)s, %(name)s) RETURNING *",
            data.model_dump(),
        ).fetchone()
    except errors.UniqueViolation:
        raise HTTPException(409, "Chat already exists")


def push_chat_history(data: PushChatHistory, conn: psycopg.Connection[DictRow]) -> DictRow | None:
    """Add one message to a chat. 404 if the chat does not exist, 409 if the message_id is repeated."""
    try:
        return conn.execute(
            "INSERT INTO chat_history (id_chat, message_id, role, content) "
            "VALUES (%(chat_id)s, %(message_id)s, %(role)s, %(content)s) RETURNING *",
            data.model_dump(),
        ).fetchone()
    except errors.ForeignKeyViolation:
        raise HTTPException(404, "Chat not found")
    except errors.UniqueViolation:
        raise HTTPException(409, "Message already stored")


def update_chat_history_with_summary(
    chat_id: str,
    summary: str,
    summarized_message_ids: list[str],
    conn: psycopg.Connection[DictRow],
) -> int:
    """Save the chat's summary and mark the messages it covers.

    `summarized_message_ids` are the LangChain message ids the summarization node
    reports (running_summary.summarized_message_ids), stored in chat_history.message_id.
    Both statements use the same connection, so they are saved or discarded together.
    Returns how many messages were marked. 404 if the chat does not exist.
    """
    updated = conn.execute(
        "UPDATE existing_chats SET summary = %s WHERE chat_id = %s RETURNING chat_id",
        (summary, chat_id),
    ).fetchone()
    if updated is None:
        raise HTTPException(404, "Chat not found")

    marked = conn.execute(
        "UPDATE chat_history SET summarized = true "
        "WHERE id_chat = %s AND message_id = ANY(%s) AND NOT summarized",
        (chat_id, summarized_message_ids),
    )
    return marked.rowcount
