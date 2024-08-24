import logging
from pathlib import Path
from typing import Optional
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from os import environ
from transformers import AutoProcessor, BlipForConditionalGeneration
from masha.image.config import image_config, Img2CaptionConfig


class ImageCaptionMeta(type):
    _instance: Optional["ImageCaption"] = None

    def __call__(cls, *args, **kwds):
        if not cls._instance:
            cls._instance = type.__call__(cls, image_config.img2caption, *args, **kwds)
        return cls._instance

    @property
    def dataRoot(cls) -> Path:
        return Path(environ.get("HUGGINGROOT", "."))

    @property
    def device(cls):
        return environ.get("DEVICE", "mps")

    def caption(cls, image: Path) -> str:
        return cls().get_answer(image)


class ImageCaption(object, metaclass=ImageCaptionMeta):

    def __init__(self, config: Img2CaptionConfig) -> None:
        self.__config = config
        self.__processor = None
        self.__model = None

    @property
    def processor(self) -> AutoProcessor:
        if not self.__processor:
            pth = self.__class__.dataRoot / self.__config.model
            self.__processor = AutoProcessor.from_pretrained(pth.as_posix())
        return self.__processor

    @property
    def model(self) -> BlipForConditionalGeneration:
        if not self.__model:
            pth = self.__class__.dataRoot / self.__config.model
            self.__model = BlipForConditionalGeneration.from_pretrained(
                pth.as_posix()
            )
        return self.__model

    def get_answer(self, image: Path):
        raw_image = Image.open(image.as_posix()).convert(
            "RGB"
        )
        exif_transpose(raw_image, in_place=True)
        processor = self.processor
        model = self.model
        device = self.__class__.device

        # unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt")

        out = model.generate(**inputs)
        result = processor.decode(out[0], skip_special_tokens=True).replace("arafed", "").strip()
        logging.info(f"CAPTION={result}")
        return result
