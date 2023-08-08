from pathlib import Path
import warnings
from hashlib import blake2b
from typing import Optional
from cachable import Cachable
from cachable.storage.meta import StorageMeta
from cachable.storage.redis import RedisStorage
from numpy import VisibleDeprecationWarning
from transformers import (
    AutoTokenizer,
    ImageToTextPipeline,
    Pipeline,
    pipeline,
)
from masha.pipelines import BasePipeline

warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)


class QuestionAnswer(Cachable):

    question: str
    context: str
    __id: Optional[str] = None
    _struct: Optional[str] = None

    def __init__(self, question, context):
        self.question = question
        self.context = context

    @property
    def storage(self) -> StorageMeta:
        return RedisStorage

    def tocache(self, res):
        self._struct = super().tocache(res)

    @property
    def id(self) -> str:
        if not self.__id:
            content = f"{self.question}:{self.context}"
            self.__id = blake2b(content.encode()).hexdigest()
        return self.__id

    @property
    def answer(self) -> Optional[str]:
        if not self.load():
            return None
        return self._struct


class Image2Text(BasePipeline):

    _tokenizer = None
    _model = None
    _pipeline: Optional[Pipeline] = None

    @property
    def tokenizer(self):
        pass

    @property
    def model(self):
        pass

    @property
    def pipeline(self):
        if not self._pipeline:
            self._pipeline = pipeline(
                task="image-to-text", model=self.modelPath.as_posix()
            )
        return self._pipeline

    def getAnswer(self, image_path: Path):
        pipe = self.pipeline
        res = pipe(images=image_path.as_posix())  # type: ignore
        return res[0].get("generated_text")
