from pathlib import Path
from typing import Optional
from masha.pipelines.text2text_generation import Text2TextGeneration
from masha.text.config import CoverLetterConfig


class CoverLetterMeta(type):
    _instance = None
    _model_path: Optional[Path] = None
    config: CoverLetterConfig

    def __call__(cls, *args, **kwds):
        if not cls._instance:
            cls._instance = type.__call__(
                cls, cls.config.model, *args, **kwds
            )
        return cls._instance

    @classmethod
    def register(cls, config: CoverLetterConfig):
        cls.config = config

    def generateText(cls, prompt: str, **kwds) -> str:
        return cls().getGeneratedResult(prompt, **kwds)


class CoverLetter(Text2TextGeneration, metaclass=CoverLetterMeta):
    pass
