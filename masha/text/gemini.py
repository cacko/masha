import logging
from typing import Optional
from gemini import Gemini as API
from pydantic import BaseModel
import requests

from masha.text.config import GeminiConfig


class GeminiResponse(BaseModel):
    content: Optional[str] = None
    links: Optional[list[str]] = None
    images: Optional[list[str]] = None


class GeminiMeta(type):
    __tokens: list[str] = []
    __instance: Optional["Gemini"] = None
    __session: Optional[requests.Session] = None

    def __call__(cls, *args, **kwds):
        if not cls.__instance:
            cls.__instance = None
        cls.__instance = type.__call__(cls, *args, **kwds)
        return cls.__instance

    def register(cls, cfg: GeminiConfig):
        cls.__tokens = cfg.cookies

    @property
    def token(cls) -> list[str]:
        return cls.__tokens

    @property
    def cookies(cls) -> dict[str, str]:
        result = {}
        for tkn in cls.__tokens:
            name, value = tkn.strip().split("=", 1)
            result[name] = value
        return result

    @property
    def session(cls):
        if not cls.__session:
            session = requests.Session()
            session.headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:123.0) Gecko/20100101 Firefox/123.0",
                "Accept": "*/*",
                "Accept-Language": "en-GB,en;q=0.9",
                "Referer": "https://gemini.google.com/",
                "X-Same-Domain": "1",
                "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
                "Origin": "https://gemini.google.com",
                "DNT": "1",
                "Connection": "keep-alive",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            }
            for tkn in cls.__tokens:
                name, value = tkn.strip().split("=", 1)
                session.cookies.update({name: value})
            cls.__session = session
        return cls.__session

    def ask(cls, query: str) -> GeminiResponse:
        return cls().do_ask(query)


class Gemini(object, metaclass=GeminiMeta):
    def __init__(self) -> None:
        try:
            self.__client = API(cookies=self.__class__.cookies)
        except Exception as e:
            logging.exception(e)
            logging.info(f"token={self.__class__.cookies}")
            raise e

    def do_ask(self, query: str) -> GeminiResponse:
        response = self.__client.generate_content(query)
        logging.debug(response.payload)
        resp = GeminiResponse(
            content=response.text, images=[str(im.url) for im in response.web_images]
        )
        return resp
