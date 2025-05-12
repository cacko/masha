import logging
from typing import Optional

from masha.image.caption import ImageCaption
from masha.image.classify import Gender as GenderClassifier, Ethnic as EthnicClassifier
from masha.image.deepface import AgeClient
from masha.image.models import Sex
from masha.name.classifiers import Ethnicity
from insightface.app import FaceAnalysis
import cv2
from pathlib import Path
from masha.image.config import image_config
from corestring import file_hash
from coreimage.transform.crop import Cropper
from coreimage.transform.upscale import Upscale
import torch
import pickle


class FaceId:

    def __init__(self, image_path: Path, out_path: Optional[Path] = None) -> None:
        self.__image_path = image_path
        self.__out_path = out_path

    @property
    def face_path(self) -> Path:
        if not self.__out_path:
            hash = file_hash(self.__image_path)
            self.__out_path = image_config.face2img.face_store / f"{hash}.face"
            logging.debug(f"generating out_path {self.__out_path}")
        return self.__out_path

    @classmethod
    def try_by_name(cls, name: str, overwrite=False):
        image_path = Path(name)
        if image_path.exists():
            return cls.create(input=image_path, overwrite=overwrite)
        name_path = image_config.face2img.face_store / f"{name}.face"
        return cls(image_path=".", out_path=name_path)

    @classmethod
    def create(cls, input: Path, output: Path = None, overwrite=True):
        try:
            obj = cls(image_path=input, out_path=output)
            assert obj.__create(overwrite=overwrite)
            return obj
        except AssertionError:
            return None

    def __cropped(self) -> Path:
        crop_path = self.path_crop
        crop = Cropper(self.__image_path, width=640, height=640, blur=False)
        return crop.crop(out=crop_path)

    def __upscaled_cropped(self, dst: Path) -> Path:
        return Upscale.upscale(src_path=self.path_crop, dst_path=dst, scale=2)

    @property
    def isGenerated(self) -> bool:
        return all(
            [
                self.face_path.exists(),
                self.path_age.exists(),
                self.path_race.exists(),
                self.path_sex.exists(),
                self.path_crop.exists(),
                self.path_caption.exists(),
            ]
        )

    def __create(self, overwrite: bool = False):
        try:
            assert not overwrite
            assert self.isGenerated
            return True
        except AssertionError:
            pass
        try:
            img_path = self.__image_path
            crop_path = self.__cropped()
            assert img_path
            app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                dtype=torch.float16,
            )
            app.prepare(ctx_id=0, det_thresh=0.1)
            image = cv2.imread(crop_path.as_posix())
            faces = app.get(image)
            face = faces.pop(0)
            faceid_embeds = torch.from_numpy(face.normed_embedding).unsqueeze(0)
            with self.face_path.open("wb") as fp:
                pickle.dump(faceid_embeds, fp)
            self.sex = self.__get_gender(self.path_upscaled_crop, face.sex)
            logging.info(f">> SEX = {self.sex}")
            self.age = self.__get_age(self.path_upscaled_crop, face.age)
            logging.info(f">> AGE = {self.age}")
            self.race = self.__get_face_ethicity(crop_path)
            logging.info(f">> RACE = {self.race.value}")
            self.caption = self.__get_caption(self.__image_path)
            logging.info(f">> CAPTION = {self.caption}")
            return True
        except AssertionError:
            raise RuntimeError("No faces detected")

    def __suffix_face_path(self, suffix: str):
        return self.face_path.parent / f"{self.face_path.stem}.{suffix}"

    @property
    def path_upscaled_crop(self) -> Path:
        res = self.__suffix_face_path("upscaled.png")
        if not res.exists():
            self.__upscaled_cropped(res)
        return res

    @property
    def path_crop(self) -> Path:
        return self.__suffix_face_path("png")

    @property
    def path_sex(self) -> Path:
        return self.__suffix_face_path("sex")

    @property
    def path_age(self) -> Path:
        return self.__suffix_face_path("age")

    @property
    def path_race(self) -> Path:
        return self.__suffix_face_path("race")

    @property
    def path_caption(self) -> Path:
        return self.__suffix_face_path("caption")

    @property
    def embeds(self):
        return pickle.loads(self.face_path.read_bytes())

    @property
    def sex(self):
        sex = self.path_sex.read_text().strip()
        return Sex[sex]

    @sex.setter
    def sex(self, value: Sex | str):
        match value:
            case str():
                self.path_sex.write_text(value[0].upper())
            case Sex():
                self.path_sex.write_text(value.name)

    @property
    def age(self) -> int:
        age = self.path_age.read_text().strip()
        return int(age)

    @age.setter
    def age(self, age: int):
        self.path_age.write_text(f"{age}")

    @property
    def race(self) -> Ethnicity:
        race = self.path_race.read_text().strip()
        return Ethnicity(race)

    @race.setter
    def race(self, value: Ethnicity):
        self.path_race.write_text(f"{value}")

    @property
    def caption(self) -> str:
        caption = self.path_caption.read_text().strip()
        return caption

    @caption.setter
    def caption(self, value: str):
        self.path_caption.write_text(f"{value}")

    def __get_face_ethicity(self, cropped: Path) -> Ethnicity:
        try:
            results = EthnicClassifier.classify(image=cropped)
            assert results
            return Ethnicity(results[0].value.lower())
        except AssertionError:
            return Ethnicity.WHITE

    def __get_gender(self, cropped: Path, default: str) -> str:
        try:
            results = GenderClassifier.classify(image=cropped)
            return results[0].value
        except Exception as e:
            logging.exception(e)
            return Sex(default)

    def __get_age(self, cropped: Path, default: int) -> str:
        try:
            return AgeClient.classify(image_path=cropped)
        except Exception as e:
            logging.exception(e)
            return default

    def __get_caption(self, img_path: Path) -> str:
        try:
            return ImageCaption.caption(image=img_path)
        except Exception as e:
            logging.exception(e)
            return ""
