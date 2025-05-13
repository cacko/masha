from masha.config import WikiConfig
from masha.pipelines.summarizaion import Summarization, Summarized
from typing import Optional


class WikiSummarizationMeta(type):
    _instance: Optional[Summarization] = None
    config: WikiConfig

    def __call__(cls, *args, **kwds):
        if not cls._instance:
            cls._instance = type.__call__(
                cls, cls.config.summarization_model, *args, **kwds
            )
        return cls._instance

    @classmethod
    def register(cls, config: WikiConfig):
        cls.config = config

    def summarization(cls, text, max_length: int = 200) -> str:
        summarized = Summarized(text, max_length=max_length)
        cls().summarize(summarized)
        assert summarized.summary
        return summarized.summary


class WikiSummarization(Summarization, metaclass=WikiSummarizationMeta):
    pass
