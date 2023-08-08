
from pathlib import Path
from typing import Optional
from masha.pipelines.token_classifier import TokenClassifier
from pydantic import BaseModel
from enum import StrEnum

from masha.text.config import SkillsConfig


class EntityGroup(StrEnum):
    TECHNICAL = "TECHNICAL"
    BUSINESS = "BUS"
    TECHNOLOGY = "TECHNOLOGY"
    SOFT = "SOFT"


class Token(BaseModel):
    entity_group: EntityGroup
    score: float
    word: str
    start: int
    end: int

    def __init__(self, **kwds):
        word = kwds.get("word")
        if all(map(str.islower, word.replace(' ', ''))):
            kwds["word"] = word.title()
        super().__init__(**kwds)

    def __eq__(self, other: 'Token') -> bool:
        return all([self.word == other.word, self.entity_group == other.entity_group])

    def __hash__(self) -> int:
        return hash(f"{self.entity_group}_{self.word}")


class SkillExtractorMeta(type):

    _instance = None
    _model_path: Optional[Path] = None
    config: SkillsConfig

    def __call__(cls, *args, **kwds):
        if not cls._instance:
            cls._instance = type.__call__(
                cls,
                cls.config.extractor_model,
                *args,
                **kwds
            )
        return cls._instance

    @classmethod
    def register(cls, config: SkillsConfig):
        cls.config = config

    def getSkills(cls, text) -> list[Token]:
        return cls().returnTokens(text)


class SkillExtractor(TokenClassifier, metaclass=SkillExtractorMeta):

    def returnTokens(self, text: str) -> list[Token]:
        tokens = [Token(**{**t, "word": text[t.get("start"):t.get("end")]}) for t in self.getGeneratedResult(text)]
        return list(set(tokens))
