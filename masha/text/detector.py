from pathlib import Path
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from pydantic import BaseModel
from typing import Optional
import masha


class Config(BaseModel):
    lang_model: str


class TextDetectorMeta(type):
    _instance: Optional['TextDetector'] = None
    _model_path: Optional[Path] = None

    def __call__(cls, *args, **kwds):
        if not cls._instance:
            cls._instance = type.__call__(cls, *args, **kwds)
        return cls._instance

    def detect(cls, text) -> str:
        return cls().get_top_language(text)


class TextDetector(object, metaclass=TextDetectorMeta):
    MAPPING = {"mk": "bg"}

    def get_top_language(self, text: str):
        try:
            match = detect(text)
            return self.MAPPING.get(match, match)
        except LangDetectException:
            return masha.chat.NATIVE_LANGUAGE
