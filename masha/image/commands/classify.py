from uuid import uuid4
import typer
from pathlib import Path
from typing_extensions import Annotated
from masha.core.image import load_image
from masha.image.classifiers.models import OBJECT, ClassifierResult
from masha.image.cli import cli
from masha.image.router import router
from rich import print, inspect
from fastapi import HTTPException, UploadFile, File
from masha.core.request import uploaded_file, make_multipart_response
from coreimage.terminal import print_term_image
from masha.image.yolo import ObjectCropper
from masha.image.classifiers import Dog, Person
from coreimage.organise import Concat
from corefile import TempPath
from PIL import Image

from masha.image.yolo.models import CropResults


def get_results(img_path) -> tuple[CropResults, list[ClassifierResult]]:
    detector = ObjectCropper()
    classified = []
    results = detector.process(
        load_image(img_path), show_only=[OBJECT.DOG, OBJECT.PERSON]
    )
    for obj in results.objects:
        match obj.cls.lower():
            case OBJECT.DOG:
                classified.append(Dog.one(image=obj.path, idx=obj.idx))
            case OBJECT.PERSON:
                classified.append(Person.one(image=obj.path, idx=obj.idx))
            case _:
                classified.append(
                    ClassifierResult(
                        label=obj.label, object_idx=obj.idx, image=obj.path, cls=obj.cls
                    )
                )
    return results, classified


@cli.command()
def detect(img_path: Annotated[Path, typer.Argument()]):
    print_term_image(image_path=img_path, height=20)
    cropper = ObjectCropper()
    results = cropper.process(load_image(img_path))
    print(results)
    print_term_image(image=results.plot_im, height=20)


# class ClassifyResponse(BaseModel):
#     objects: Optional[list[ClassifyResult]] = None
#     age: Optional[list[ClassifyResult]] = None
#     gender: Optional[list[ClassifyResult]] = None
#     attraction: Optional[list[ClassifyResult]] = None
#     ethnicity: Optional[list[ClassifyResult]] = None

#     def response(self):
#         result = [*self.objects]
#         try:
#             assert len(self.age)
#             result.append(self.age.pop(0))
#             assert len(self.gender)
#             result.append(self.gender.pop(0))
#             assert len(self.attraction)
#             result.append(self.attraction.pop(0))
#             assert len(self.ethnicity)
#             result.append(self.ethnicity.pop(0))
#         except AssertionError:
#             return result
#         return result


@router.post("/classify")
async def api_classify(
    file: Annotated[UploadFile, File()],
):
    tmp_path = await uploaded_file(file)
    results, classified = get_results(tmp_path)
    annotated_path = TempPath(f"annotated_{uuid4()}.jpg")
    results.save(annotated_path)
    response = make_multipart_response(
        image_path=annotated_path, message="\n".join([x.result for x in classified])
    )
    annotated_path.unlink(missing_ok=True)
    return response


@cli.command()
def classify(img_path: Annotated[Path, typer.Argument()]):
    print_term_image(image_path=img_path, height=20)
    results, classified = get_results(img_path)
    print_term_image(image=results.plot_im, height=40)
    concat_path = TempPath("concat")
    crops, _ = Concat(dst=concat_path).concat_from_paths(
        paths=[d.path for d in results.objects]
    )
    print_term_image(image_path=crops)
    for x in classified:
        print(x.result)
