from pydantic import BaseModel
from typing import Optional


class Payload(BaseModel):
    message: str
    source: Optional[str] = None
    lang: Optional[str] = None
    detect_lang: Optional[bool] = None
