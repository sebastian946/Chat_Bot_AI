
from pydantic import BaseModel


class ModelInput(BaseModel):
    user_name: str
    prompt_user: str
    topic:str
