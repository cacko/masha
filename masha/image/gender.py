
import logging
from masha.image.config import ClassifyConfig
from masha.pipelines.image_clasify import ImageClassify
from pathlib import Path
from typing import Optional
from masha.pipelines.image_clasify.models import ClassifyResult
from pydantic import BaseModel


class Config(BaseModel):
    model: str
    tokenizer: str


class GenderMeta(type):

    _instance: Optional[ImageClassify] = None
    config: Optional[ClassifyConfig] = None

    def __call__(cls, *args, **kwds):
        if not cls._instance:
            assert cls.config
            cls._instance = type.__call__(
                cls,
                cls.config.gender_model,
                cls.config.gender_model,
                *args, **kwds)
        return cls._instance

    def register(cls, config: ClassifyConfig):
        logging.debug(config)
        cls.config = config

    def classify(cls, image: Path, threshold=0.4) -> list[ClassifyResult]:
        return cls().getAnswer(image, threshold)


class GenderImageClassifier(ImageClassify, metaclass=GenderMeta):
    pass

