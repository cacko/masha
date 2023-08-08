
from pathlib import Path
from typing import Optional
from masha.pipelines.text_generation import TextGeneration
from masha.text.config import GeneratorConfig


class TextGeneratorMeta(type):

    _instance = None
    _model_path: Optional[Path] = None
    config: GeneratorConfig

    def __call__(cls, *args, **kwds):
        if not cls._instance:
            cls._instance = type.__call__(
                cls,
                cls.config.model,
                *args,
                **kwds
            )
        return cls._instance

    @classmethod
    def register(cls, config: GeneratorConfig):
        cls.config = config

    def generateText(cls, prefix) -> str:
        return cls().getGeneratedResult(prefix)


class TextGenerator(TextGeneration, metaclass=TextGeneratorMeta):
    pass
