from pydantic import BaseModel

class CreateChat(BaseModel):
    chat_id: str
    role: str
    content: str
    created_at: str

class OutputChat(BaseModel):
    chat_id: str
    response_message: str