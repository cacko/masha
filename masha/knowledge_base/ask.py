from typing import Optional
from masha.core.models import WorkerReponse
from masha.knowledge_base.wiki import Wiki
from masha.knowledge_base.wolfram import Wolfram
from masha.pipelines.question_answering import QuestionAnswer, QuestionAnswering
from masha.config import WikiConfig


class AskMeta(type):

    _instances: dict["str", "Ask"] = {}
    _qa: Optional[QuestionAnswering] = None

    def __call__(cls, question, *args, **kwds):
        if question not in cls._instances:
            cls._instances[question] = type.__call__(cls, question, *args, **kwds)
        return cls._instances[question]

    @classmethod
    def register(cls, config: WikiConfig):
        cfg = config
        cls._qa = QuestionAnswering(
            cfg.qa_model,
            cfg.qa_model,
        )

    def ask(cls, question: str) -> Optional[WorkerReponse]:
        return cls(question).get_answer()


class Ask(object, metaclass=AskMeta):

    __question: str

    def __init__(self, question: str) -> None:
        self.__question = question

    def get_answer(self) -> Optional[WorkerReponse]:
        try:
            resp = self.get_wolfram()
            if resp:
                return resp
            wiki = Wiki(self.__question)
            content = wiki.content
            if not content:
                return WorkerReponse()
            assert Ask._qa
            answer = Ask._qa.getAnswer(QuestionAnswer(self.__question, content))
            return WorkerReponse(message=answer.answer)
        except (FileNotFoundError, StopIteration, AssertionError):
            return None

    def get_wolfram(self):
        try:
            return Wolfram.answer(self.__question)
        except (FileNotFoundError, StopIteration):
            return None
