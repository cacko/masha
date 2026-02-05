from transformers import AutoModelForCausalLM, AutoTokenizer
from masha.code.config import SummarizationConfig
from masha.pipelines.summarizaion import Summarization, Summarized
from masha.core import perftime
from nltk.tokenize import WordPunctTokenizer
from typing import Optional


class CodeSummarized(Summarized):

    def __init__(self, context, min_length=50, max_length=80):
        tokenized_list = WordPunctTokenizer().tokenize(context)
        context = ' '.join(tokenized_list[:100])
        super().__init__(context, min_length, max_length)

    def load(self) -> bool:
        return False

    def tocache(self, res):
        self._struct = res

    @property
    def summary(self) -> str:
        assert self._struct
        return self._struct


class CodeSummarizationMeta(type):
    _instance: Optional[Summarization] = None
    config: Optional[SummarizationConfig] = None

    def __call__(cls, *args, **kwds):
        if not cls._instance:
            assert cls.config
            cls._instance = type.__call__(
                cls,
                cls.config.model,
                *args,
                **kwds
            )
        return cls._instance

    @classmethod
    def register(cls, config: SummarizationConfig):
        cls.config = config

    def summarization(cls, text) -> str:
        summarized = CodeSummarized(text)
        cls().summarize(summarized)
        return summarized.summary


class CodeSummarization(Summarization, metaclass=CodeSummarizationMeta):

    @property
    def tokenizer(self):
        if not self._tokenizer:
            with perftime("load summarization tokenizer"):
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizerPath.as_posix(),
                    legacy=False
                )
        return self._tokenizer

    @property
    def model(self):
        if not self._model:
            with perftime("load summarization model"):
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.modelPath.as_posix(),
                )
        return self._model
