from typing import Optional
from corefile import TempPath
from pydantic import BaseModel
import typer
from pathlib import Path
from typing_extensions import Annotated
from masha.image.classify import Age, Attraction, Classifier, Ethnic, Gender
from masha.image.cli import cli
from masha.image.deepface import AgeClient, RaceClient
from masha.image.router import router
from rich import print
from fastapi import UploadFile, File
from masha.core.request import uploaded_file
from coreimage.terminal import print_term_image
from coreimage.transform.crop import Cropper
from masha.image.yolo import detect_objects
from masha.image.dog import get_dog_breed


from masha.pipelines.image_clasify.models import ClassifyResult


@cli.command()
def dog(img_path: Annotated[Path, typer.Argument()]):
    print_term_image(image_path=img_path, height=20)
    get_dog_breed(image=img_path)


@cli.command()
def detect(img_path: Annotated[Path, typer.Argument()]):
    print_term_image(image_path=img_path, height=20)
    detect_objects(img_path=img_path)


class ClassifyResponse(BaseModel):
    objects: Optional[list[ClassifyResult]] = None
    age: Optional[list[ClassifyResult]] = None
    gender: Optional[list[ClassifyResult]] = None
    attraction: Optional[list[ClassifyResult]] = None
    ethnicity: Optional[list[ClassifyResult]] = None

    def response(self):
        result = [*self.objects]
        try:
            assert len(self.age)
            result.append(self.age.pop(0))
            assert len(self.gender)
            result.append(self.gender.pop(0))
            assert len(self.attraction)
            result.append(self.attraction.pop(0))
            assert len(self.ethnicity)
            result.append(self.ethnicity.pop(0))
        except AssertionError:
            return result
        return result


@router.post("/classify")
async def api_classify(
    file: Annotated[UploadFile, File()],
):
    tmp_path = await uploaded_file(file)
    res = ClassifyResponse(
        objects=Classifier.classify(tmp_path),
        age=Age.classify(tmp_path),
        gender=Gender.classify(tmp_path),
        attraction=Attraction.classify(tmp_path),
        ethnicity=Ethnic.classify(tmp_path),
    )
    return {"response": res.response()}


@router.post("/dog")
async def api_dog(
    file: Annotated[UploadFile, File()],
):
    tmp_path = await uploaded_file(file)
    res = get_dog_breed(image=tmp_path)
    return {"response": res}


@cli.command()
def classify(img_path: Annotated[Path, typer.Argument()]):
    print_term_image(image_path=img_path, height=20)
    tmp_path = TempPath(img_path.name)
    cropper = Cropper(img_path)
    crop_path = cropper.crop(out=tmp_path)
    print(RaceClient.classify(image_path=crop_path))
    print(AgeClient.classify(crop_path))
    res = ClassifyResponse(
        objects=Classifier.classify(img_path, threshold=0),
        age=Age.classify(img_path),
        gender=Gender.classify(img_path),
        attraction=Attraction.classify(img_path),
        ethnicity=Ethnic.classify(img_path),
    )
    print(res.response())
