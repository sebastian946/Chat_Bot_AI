from pydantic import BaseModel

class CreateChat(BaseModel):
    chat_id: str
    name: str

class PushChatHistory(BaseModel):
    chat_id: str
    role: str
    content: str
    created_at: str
    message_id: str | None = None   # LangChain message id; needed to mark the message as summarized

class OutputChat(BaseModel):
    chat_id: str
    response_message: str