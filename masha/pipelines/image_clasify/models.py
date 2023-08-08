from pydantic import BaseModel


class ClassifyResult(BaseModel):
    label: str
    score: float
    cls: str
