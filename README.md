# Chat Bot AI

A topic-expert chatbot built with LangGraph and Claude. The user picks a topic
(psychology, engineering, medicine or history) and talks to an expert on it.
Conversations are stored in PostgreSQL, long conversations are summarized
automatically, and a conversation can be resumed after the app restarts.

It can be used from an interactive console or through a FastAPI HTTP API.

## How it works

Each question runs through this graph:

```
init → summarization → save_summary → psychology | engineering | medicine | history
```

| Node | What it does |
|---|---|
| `init` | On the first question of a conversation, loads its summary and pending messages from Postgres. Greets a new user, or welcomes back a returning one. |
| `summarization` | When the conversation grows past a token threshold, summarizes the older messages. The model then receives the summary plus the recent messages instead of the full history. |
| `save_summary` | Stores a new summary in Postgres and marks the messages it covers. |
| expert node | Answers as the expert of the chosen topic and stores the question and the answer. |

Things worth knowing:

- **Messages are never deleted.** Summarized messages are only marked as
  `summarized`; the full history stays available.
- **The current question is never summarized.** The summary only covers
  previous turns, so the question always reaches the model word for word.
- **User input is wrapped in `<user_input>` tags** before it reaches the model,
  and the system prompt tells the model that nothing inside those tags can
  change its role or rules. The database stores the original text.
- **Two layers of memory.** While the process runs, a LangGraph in-memory
  checkpointer keeps each conversation. Postgres is the durable copy, read once
  per conversation after a restart.

The model and the summarization thresholds are set at the top of
[llm/graph.py](llm/graph.py) (`MODEL_NAME`, `max_tokens`,
`max_tokens_before_summary`, `max_summary_tokens`).

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Docker Desktop (for PostgreSQL)
- An Anthropic API key

## Setup

1. Install the dependencies:

   ```powershell
   uv sync
   ```

2. Create a `.env` file in the project root:

   ```
   ANTHROPIC_API_KEY=your-api-key
   POSTGRES_USER=chat_user
   POSTGRES_PASSWORD=choose-a-password
   POSTGRES_DB_NAME=chat_bot
   POSTGRES_PORT=5433
   REDIS_HOST=localhost
   REDIS_PORT=6379
   ```

   | Variable | Notes |
   |---|---|
   | `POSTGRES_PORT` | Port published on your machine. Use a free one: if PostgreSQL is already installed locally it usually holds 5432. |
   | `POSTGRES_HOST` | Optional, defaults to `localhost`. |
   | `REDIS_HOST`, `REDIS_PORT` | Required by the settings, but Redis is not used yet. |

3. Start the database:

   ```powershell
   docker compose up -d db
   docker compose ps
   ```

   Wait until the status shows `healthy`. The tables are created automatically
   the first time the console or the API starts.

## Running the console

```powershell
uv run python -m llm.graph
```

It asks for your name and a topic, then answers questions until you type
`exit`. After each answer it prints how many messages are stored, how many
were sent to the model, and the current summary if there is one.

The console always uses the conversation `console`, so opening it again
continues where you left off.

## Running the API

```powershell
uv run uvicorn main:app --reload
```

Interactive documentation is available at <http://localhost:8000/docs>.

| Method | Path | Description | Responses |
|---|---|---|---|
| `GET` | `/health` | Liveness check | 200 |
| `POST` | `/chats` | Create a chat | 201, 409 if it already exists |
| `GET` | `/chats/{chat_id}/history` | Messages of a chat, oldest first | 200, 404 if the chat does not exist |
| `POST` | `/chats/{chat_id}/messages` | Ask the expert a question | 200, 400 for an invalid topic, 503 if the database is unreachable |

Create a chat:

```json
POST /chats
{"chat_id": "ana-1", "name": "Ana's history chat"}
```

Ask a question (the chat is created on its first message if it does not exist):

```json
POST /chats/ana-1/messages
{"user_name": "Ana", "prompt_user": "Who was Simón Bolívar?", "topic": "history"}
```

```json
{"chat_id": "ana-1", "response_message": "Simón Bolívar was ..."}
```

`topic` must be one of `psychology`, `engineering`, `medicine` or `history`.

## Database

| Table | Columns | Purpose |
|---|---|---|
| `existing_chats` | `chat_id`, `name`, `summary` | One row per conversation, with its latest summary. |
| `chat_history` | `id`, `id_chat`, `message_id`, `role`, `content`, `summarized`, `created_at` | One row per message. `summarized` is true once the summary covers it. |

`message_id` is the id the message has inside the graph. It is what links a
summary to the rows it covers.

To inspect the data:

```powershell
docker compose exec db psql -U <POSTGRES_USER> -d <POSTGRES_DB_NAME>
```

```sql
SELECT chat_id, name, summary FROM existing_chats;
SELECT id, role, summarized, left(content, 40) FROM chat_history ORDER BY id;
```

## Tests

```powershell
uv run pytest -m "not real_llm"
```

This runs everything that does not call the real model. Nothing is spent and
nothing is left in the database.

| Kind | Where | How it works |
|---|---|---|
| Graph and memory | [tests/test_memory.py](tests/test_memory.py) | The model is replaced by a fake that records what it receives. |
| Database | [tests/test_postgres.py](tests/test_postgres.py) | Runs against the real Postgres inside a transaction that is rolled back. |
| API routes | [tests/test_routes.py](tests/test_routes.py) | FastAPI test client; the graph is replaced by a stub. |

Tests marked `db` are skipped when Postgres is not reachable, so `skipped`
in the output means the database is down.

To check the full flow against the real model (this calls the Anthropic API
and spends tokens):

```powershell
uv run pytest -m real_llm
```

It tells the bot a fact, pushes the conversation past the summarization
threshold, and checks the bot still remembers the fact. It uses its own chat
and deletes it afterwards.

## Project structure

```
config/settings.py     Settings read from .env
db/postgress.py        Connection, table creation and queries
llm/graph.py           The LangGraph flow, prompts, memory and the console
models/                Request and response models
routes/routes.py       API routes
main.py                FastAPI app
tests/                 Automated tests
docker-compose.yml     PostgreSQL service
Dockerfile             Image for the API
```

## Known limitations

- **No authentication.** Anyone who knows a `chat_id` can read and write that
  conversation. Add authentication before exposing the API.
- **Single process.** The in-process memory is not shared between instances;
  run one `uvicorn` worker.
- **The topic is sent with every message.** Changing it in the middle of a
  chat makes a different expert answer over the same conversation.
- **Summaries can lose detail.** Right after a summary the model mostly sees
  the summary, so a small `max_summary_tokens` can drop specific facts.
