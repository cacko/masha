import warnings
from hashlib import blake2b
from typing import Optional
from cachable import Cachable
from cachable.storage.meta import StorageMeta
from cachable.storage.redis import RedisStorage
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Pipeline,
    pipeline,
)
from masha.pipelines import BasePipeline



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


class QuestionAnswering(BasePipeline):

    _tokenizer = None
    _model = None
    _pipeline: Optional[Pipeline] = None

    @property
    def tokenizer(self):
        if not self._tokenizer:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizerPath.as_posix()
            )
        return self._tokenizer

    @property
    def model(self):
        if not self._model:
            self._model = AutoModelForQuestionAnswering.from_pretrained(
                self.modelPath.as_posix()
            )
        return self._model

    @property
    def pipeline(self):
        if not self._pipeline:
            self._pipeline = pipeline(
                task="question-answering", model=self.model, tokenizer=self.tokenizer
            )
        return self._pipeline

    def getAnswer(self, answer: QuestionAnswer) -> QuestionAnswer:
        if not answer.load():
            pipe = self.pipeline
            res = pipe(question=answer.question, context=answer.context)  # type: ignore
            answer.tocache(res["answer"])
        return answer
