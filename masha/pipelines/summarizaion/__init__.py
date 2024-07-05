from cachable import Cachable
from cachable.storage.meta import StorageMeta
from cachable.storage.redis import RedisStorage
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from masha.core import perftime
from masha.pipelines import BasePipeline
from hashlib import blake2b
import warnings
from typing import Optional, Any


warnings.filterwarnings("ignore")


class Summarized(Cachable):
    context: Optional[str] = None
    min_length = 50
    max_length = 200
    __id: Optional[str] = None
    _struct: Optional[str] = None

    def __init__(self, context, min_length=50, max_length=200):
        self.context = context
        self.min_length = min_length
        self.max_length = max_length

    @property
    def storage(self) -> StorageMeta:
        return RedisStorage

    def tocache(self, res: Any):
        self._struct = super().tocache(res)

    @property
    def id(self) -> str:
        if not self.__id:
            content = f"{self.context}{self.min_length}{self.max_length}"
            self.__id = blake2b(content.encode()).hexdigest()
        return self.__id

    @property
    def summary(self) -> Optional[str]:
        if not self.load():
            return None
        return self._struct


class Summarization(BasePipeline):
    _tokenizer = None
    _model = None
    _pipeline = None

    @property
    def tokenizer(self):
        if not self._tokenizer:
            with perftime("load summarization tokenizer"):
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizerPath.as_posix()
                )
                self._tokenizer.model_max_length = 1024
        return self._tokenizer

    @property
    def model(self):
        if not self._model:
            with perftime("load summarization model"):
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.modelPath.as_posix()
                )
                self._model.eval()
        return self._model

    @property
    def pipeline(self):
        if not self._pipeline:
            self._pipeline = pipeline(
                task="summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.__class__.device,
            )
        return self._pipeline

    def summarize(self, summarized: Summarized) -> Optional[str]:
        if not summarized.load():
            assert summarized.context
            with perftime(f"summarizing {len(summarized.context)} characters"):
                res = self.pipeline(
                    summarized.context,
                    min_length=summarized.min_length,
                    max_length=summarized.max_length,
                )
                assert isinstance(res, list)
                result = res[0]
                summarized.tocache(result["summary_text"])  # type: ignore
        return None
