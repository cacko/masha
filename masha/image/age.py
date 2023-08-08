
import logging
from masha.image.config import ImageConfig
from masha.image.config import ClassifyConfig
from masha.pipelines.image_clasify import ImageClassify
from pathlib import Path
from typing import Optional
from masha.pipelines.image_clasify.models import ClassifyResult
from pydantic import BaseModel
from random import randrange

class Config(BaseModel):
    model: str
    tokenizer: str


class AgeMeta(type):

    _instance: Optional[ImageClassify] = None
    config: Optional[ClassifyConfig] = None

    def __call__(cls, *args, **kwds):
        if not cls._instance:
            assert cls.config
            cls._instance = type.__call__(
                cls,
                cls.config.age_model,
                cls.config.age_model,
                *args, **kwds)
        return cls._instance

    def register(cls, config: ClassifyConfig):
        cls.config = config

    def classify(cls, image: Path, threshold=0.4) -> list[ClassifyResult]:
        return cls().getAnswer(image, threshold)
    
    def label2age(cls, res: ClassifyResult):
        return randrange(*[int(x) for x in res.label.split("-", 2)])


class AgeClassifier(ImageClassify, metaclass=AgeMeta):
    pass

