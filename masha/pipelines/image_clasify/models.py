from pydantic import BaseModel
from stringcase import titlecase


class ClassifyResult(BaseModel):
    value: str
    score: float
    cls: str
    
    @property
    def result(self):
        return f"{self.cls}={self.value}"
