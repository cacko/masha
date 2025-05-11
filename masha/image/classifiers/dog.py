from gzip import FCOMMENT
from pathlib import Path
from typing import Any
from corefile import TempPath
import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
import torch
from stringcase import titlecase
from masha.core.image import load_image
from masha.image.config import image_config
from masha.image.yolo.object_cropper import ObjectCropper
from masha.pipelines.image_clasify.models import ClassifyResult
from .models import ClassifierResult, OBJECT
from uuid import uuid4

DATA_ROOT = Path(os.environ.get("HUGGINGROOT", "."))


class DogBreedMeta(type):

    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        return type.__call__(cls, *args, **kwds)

    @property
    def model_path(cls):
        return DATA_ROOT / image_config.classify.dog_breed_model

    def one(cls, image: Path, idx: int):
        return cls().classify(image, object_idx=idx)

    def batch(cls, image: Path):
        return cls().crop_and_classify(image)


class Dog(object, metaclass=DogBreedMeta):

    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.__class__.model_path.as_posix()
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            self.__class__.model_path.as_posix()
        )

    @property
    def hasDogs(self):
        return len(self.dogs) > 0

    @property
    def dogs(self):
        return self.objects.forClass("dog")

    @property
    def plot_im(self):
        return self.objects.plot_im

    @property
    def plot_path(self):
        plot_path = TempPath(f"plot_{uuid4()}.jpg")
        if not plot_path.exists():
            cv2.imwrite(plot_path.as_posix(), self.plot_im)
        return plot_path

    def crop_and_classify(self, image: Path):
        try:
            self.objects = ObjectCropper().process(
                load_image(image), show_only=[OBJECT.DOG]
            )
            assert self.hasDogs
            return [self.classify(image=d.path, object_idx=d.idx) for d in self.dogs]
        except AssertionError:
            raise ValueError("No dogs found")

    def classify(self, image: Path, object_idx: int) -> ClassifierResult:
        img = load_image(image.as_posix())
        inputs = self.image_processor(images=img, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits

        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

        predictions = {str(i): round(probs[i], 3) for i in range(len(probs))}
        predicted_class_idx = logits.argmax(-1).item()
        return ClassifierResult(
            label=[
                ClassifyResult(
                    value=titlecase(self.model.config.id2label[predicted_class_idx]),
                    score=float(predictions[str(predicted_class_idx)]),
                    cls="Breed",
                )
            ],
            cls=self.__class__.__name__,
            image=image,
            object_idx=object_idx,
        )
