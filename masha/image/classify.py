
from masha.image.config import image_config
from masha.image.config import ClassifyConfig
from masha.pipelines.image_clasify import ImageClassify
from pathlib import Path
from typing import Optional
from masha.pipelines.image_clasify.models import ClassifyResult
from pydantic import BaseModel


class Config(BaseModel):
    model: str
    tokenizer: str


class ClassifyMeta(type):

    _instance: Optional[ImageClassify] = None

    def __call__(cls, *args, **kwds):
        if not cls._instance:
            cls._instance = type.__call__(
                cls,
                image_config.classify,
                *args, **kwds)
        return cls._instance

    def classify(cls, image: Path, threshold=0.1) -> list[ClassifyResult]:
        return cls().getAnswer(image, threshold)

class Classifier(ImageClassify, metaclass=ClassifyMeta):
    
    def __init__(self, config: ClassifyConfig):
        super().__init__(config.model, config.tokenizer)

class Age(ImageClassify, metaclass=ClassifyMeta):
    
    def __init__(self, config: ClassifyConfig):
        super().__init__(config.age_model, config.age_model)

class Gender(ImageClassify, metaclass=ClassifyMeta):
    
    def __init__(self, config: ClassifyConfig):
        super().__init__(config.gender_model, config.gender_model)
        
class Ethnic(ImageClassify, metaclass=ClassifyMeta):
    
    def __init__(self, config: ClassifyConfig):
        super().__init__(config.ethnicity_model, config.ethnicity_model)
        
class Attraction(ImageClassify, metaclass=ClassifyMeta):
    
    def __init__(self, config: ClassifyConfig):
        super().__init__(config.attraction_model, config.attraction_model)