from pathlib import Path
from pydantic import BaseModel, Extra
from typing import Optional


class PostData(BaseModel, extra=Extra.ignore):
    message: Optional[str] = None
    source: Optional[str] = None
    lang: Optional[str] = None


class WorkerReponse(BaseModel, extra=Extra.ignore):
    message: Optional[str] = None
    image_url: Optional[str | Path] = None
