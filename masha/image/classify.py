from masha.core.image import load_image
from masha.image.config import image_config
from masha.image.config import ClassifyConfig
from masha.pipelines.image_clasify import ImageClassify
from pathlib import Path
from typing import Optional
from masha.pipelines.image_clasify.models import ClassifyResult
from pydantic import BaseModel
from corefile import TempPath
from coreimage.transform.crop import Cropper
from ultralytics import YOLO


class Config(BaseModel):
    model: str
    tokenizer: str

                                        
class ClassifyMeta(type):

    _instance: Optional[ImageClassify] = None

    def __call__(cls, *args, **kwds):
        if not cls._instance:
            cls._instance = type.__call__(cls, image_config.classify, *args, **kwds)
        return cls._instance

    def classify(cls, image: Path, threshold=0.1, **kwargs) -> list[ClassifyResult]:
        return cls().getAnswer(image, threshold, **kwargs)


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

    def getAnswer(self, image: Path, threshold=0.2, **kwargs):
        crop_path = image
        if not kwargs.get("cropped"):
            tmp_path = TempPath(image.name)
            cropper = Cropper(image)
            crop_path = cropper.crop(out=tmp_path)
        model = YOLO(self.modelPath)
        res = model.predict(source=crop_path)[0]
        return [
            ClassifyResult(
                value=res.names[res.probs.top1], score=res.probs.top1conf, cls="Ethnic"
            )
        ]


class Attraction(ImageClassify, metaclass=ClassifyMeta):

    def __init__(self, config: ClassifyConfig):
        super().__init__(config.attraction_model, config.attraction_model)
        
class Expression(ImageClassify, metaclass=ClassifyMeta):

    def __init__(self, config: ClassifyConfig):
        super().__init__(config.expression_model, config.expression_model)
