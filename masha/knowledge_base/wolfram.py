import logging
from hashlib import blake2s

from cachable import Cachable
from cachable.storage.meta import StorageMeta
from cachable.storage.redis import RedisStorage
from corestring import titlecase
import rich
from wolframalpha import Client
from masha.config import WolframConfig

from masha.core.models import WorkerReponse


class WolframClient(Cachable):

    __client: Client
    __question: str

    @property
    def storage(self) -> StorageMeta:
        return RedisStorage

    def __init__(self, app_id: str, question: str) -> None:
        self.__client = Client(app_id)
        self.__question = question

    @property
    def id(self):
        h = blake2s()
        h.update(self.__question.encode())
        return h.hexdigest()

    @property
    def content(self):
        parts = self.__question.split("|")
        payload = {
            "input": parts[0].strip(),
        }
        if len(parts) > 1:
            payload["params"] = [  # type: ignore
                ("assumption", f"*C.{parts[0].strip()}-_*{titlecase(x)}-")
                for x in parts[1:]
            ]
            payload["params"].append(("assumtion", "DateOrder_**Year.Month.Day--"))
        self._struct = self.__client.query(**payload)
        self.tocache(self._struct)
        return self._struct


class WolframMeta(type):

    _instance = None
    _config: WolframConfig

    def __call__(cls, *args, **kwds):
        if not cls._instance:
            cls._instance = type.__call__(cls, *args, **kwds)
        return cls._instance

    def register(cls, config: WolframConfig):
        cls._config = config

    def query(cls, query) -> WorkerReponse:
        return cls().do_query(query)

    def answer(cls, query) -> WorkerReponse:
        return cls().get_answer(query)


class Wolfram(object, metaclass=WolframMeta):
    def get_answer(self, query: str):
        res = WolframClient(Wolfram._config.app_id, question=query).content
        if not res.success:
            return None
        answer = next(res.results)
        img_url = None
        if hasattr(answer, "subpod"):
            if hasattr(answer.subpod, "imagesource"):
                img_url = answer.subpod.imagesource
        return WorkerReponse(message=answer.text, image_url=img_url)

    def do_query(self, query: str):
        res = WolframClient(Wolfram._config.app_id, question=query).content
        if not res.success:
            return None
        texts = []
        img_url = None
        for pod in res.pods:
            try:
                nsubpods = pod.get("@numsubpods", 0)
                assert nsubpods > 0
                subpods = pod.subpods if nsubpods > 1 else [pod.subpod]
                for sub in subpods:
                    rich.print(sub)
                    sub = pod.subpod
                    assert sub
                    if sub.plaintext:
                        texts.append(sub.plaintext)
                    assert hasattr(sub, "imagesource")
                    try:
                        img_url = sub.get("imagesource")
                    except Exception as e:
                        img_url = None
                        logging.exception(e)
            except AssertionError: 
                pass

        return WorkerReponse(message="\n".join(texts), image_url=img_url)
