import logging
from pathlib import Path
from typing import Any
from corefile import TempPath
import cv2
import os
from masha.core.image import load_image
from masha.image.yolo.object_cropper import ObjectCropper
from uuid import uuid4
from masha.image.classify import Age, Expression, Ethnic, Gender
from coreimage.transform.crop import Cropper

from masha.pipelines.image_clasify.models import ClassifyResult
from .models import OBJECT, ClassifierResult


DATA_ROOT = Path(os.environ.get("HUGGINGROOT", "."))


class PersonMeta(type):

    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        return type.__call__(cls, *args, **kwds)

    def one(cls, image: Path, idx: int) -> ClassifierResult:
        return cls().classify(image, object_idx=idx)

    def batch(cls, image: Path) -> list[ClassifierResult]:
        return cls().crop_and_classify(image)


class Person(object, metaclass=PersonMeta):

    def __init__(self):
        pass

    @property
    def hasPeople(self):
        return len(self.people) > 0

    @property
    def people(self):
        return self.objects.forClass("person")

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
                load_image(image), show_only=[OBJECT.PERSON]
            )
            assert self.hasPeople
            return [self.classify(image=d.path, object_idx=d.idx) for d in self.people]
        except AssertionError:
            raise ValueError("No dogs found")

    def classify(self, image: Path, object_idx: int):
        tmp_path = TempPath(image.name)
        try:
            cropper = Cropper(image)
            crop_path = cropper.crop(out=tmp_path)
            ages: list[ClassifyResult] = Age.classify(crop_path)
            genders = Gender.classify(crop_path)
            ethnos = Ethnic.classify(crop_path)
            expression: list[ClassifyResult] = Expression.classify(crop_path)
            return ClassifierResult(
                label=list(map(lambda c: c.pop(0), [genders, ages, ethnos, expression])),
                cls=self.__class__.__name__,
                image=image,
                object_idx=object_idx,
            )
        except Exception as e:
            return ClassifierResult(
                label=None,
                cls=self.__class__.__name__,
                image=image,
                object_idx=object_idx,
            )
