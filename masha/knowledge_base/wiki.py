import logging
from hashlib import blake2b
from typing import Optional
import cachable
from pydantic import BaseModel, Extra
import wikipediaapi
from wikipediaapi import WikipediaPage
from cachable import Cachable
from cachable.storage.meta import StorageMeta
from cachable.storage.redis import RedisStorage
from cachable.ua import UA

wikipediaapi.log.setLevel(level=wikipediaapi.logging.WARN)


wikipedia = wikipediaapi.Wikipedia(language="en", user_agent=UA.random)


class WikiStruct(BaseModel, extra=Extra.ignore):
    summary: str
    content: str


class PageItem(BaseModel, extra=Extra.ignore):
    pageid: int
    ns: int
    title: str
    index: int


class QueryItem(BaseModel, extra=Extra.ignore):
    pages: Optional[dict[str, PageItem]] = None


class QueryResponse(BaseModel, extra=Extra.ignore):
    query: Optional[QueryItem] = None


class Wiki(Cachable):

    __query = None
    __id = None
    _struct: WikiStruct

    @property
    def storage(self) -> StorageMeta:
        return RedisStorage

    def __init__(self, name: str):
        if not name:
            raise ValueError
        self.__query = name

    def search(self) -> WikipediaPage:
        try:
            assert self.__query
            page = wikipedia.page(self.__query)
            assert page.exists()
            return page
        except AssertionError:
            raise FileNotFoundError

    def generate(self) -> Optional[WikiStruct]:
        try:
            page = self.search()
            self._struct = WikiStruct(
                summary=page.summary,
                content=page.text,
            )
            self.tocache(self._struct)
        except Exception as e:
            logging.exception(e)
            raise FileNotFoundError
        return None

    @property
    def id(self):
        if not self.__id:
            h = blake2b(digest_size=20)
            assert self.__query
            h.update(self.__query.encode())
            self.__id = h.hexdigest()
        return self.__id

    @property
    def summary(self) -> str:
        if not self.load():
            self.generate()
        return self._struct.summary

    @property
    def content(self) -> str:
        if not self.load():
            self.generate()
        return self._struct.content
