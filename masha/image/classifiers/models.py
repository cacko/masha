from typing import Optional
from pydantic import BaseModel
from pathlib import Path
from enum import StrEnum

from masha.pipelines.image_clasify.models import ClassifyResult


class ClassifierResult(BaseModel):
    object_idx: int
    image: Path
    cls: str
    label: Optional[str | list[ClassifyResult]] = None

    def get_labels_result(self):
        for label in self.label:
            match label:
                case ClassifyResult():
                    yield label.result
                case _:
                    return label

    @property
    def title(self):
        idx = f" #{self.object_idx}" if self.object_idx else ""
        return f"{self.cls}{idx}"

    @property
    def result(self):
        if isinstance(self.label, list):
            props = [f"{l}" for l in self.get_labels_result()]
            return f"{self.title} - {', '.join(props)}"
        return self.title


class OBJECT(StrEnum):
    DOG = "dog"
    PERSON = "person"
