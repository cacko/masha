from pathlib import Path
from google import genai
from google.genai import types


import logging
from typing import Optional
from pydantic import BaseModel
import rich


class Recipe(BaseModel):
    name: str
    teams: list[str]
    goals: int
    games: int


from masha.text.config import GenaiConfig


class GeminiResponse(BaseModel):
    content: Optional[str] = None
    links: Optional[list[str]] = None
    images: Optional[list[str]] = None


class GeminiMeta(type):
    __instance: Optional["Gemini"] = None

    def __call__(cls, *args, **kwds):
        if not cls.__instance:
            cls.__instance = None
        cls.__instance = type.__call__(cls, *args, **kwds)
        return cls.__instance

    def register(cls, cfg: GenaiConfig):
        cls.__cfg = cfg

    @property
    def api_key(cls) -> str:
        return cls.__cfg.api_key

    @property
    def model_name(cls) -> str:
        return cls.__cfg.model

    def ask(
        cls, query: str, source: Optional[str] = None, fpath: Optional[Path] = None
    ) -> GeminiResponse:
        return cls().do_ask(query=query, source=source, fpath=fpath)

    def ask_image(cls, img_path: Path, query: str):
        return cls().do_ask_image(img_path=img_path, query=query)

    def ask_json(cls, query: str, source: Optional[str] = None) -> GeminiResponse:
        return cls().do_ask_json(query=query, source=source)
    
    def ask_schema(
        cls, query: str, schema: BaseModel
    ):
        return cls().do_ask_schema(query=query, schema=schema)


class Gemini(object, metaclass=GeminiMeta):

    __chat: dict[str, any] = {}

    def __init__(self) -> None:
        try:
            self.__client = genai.Client(api_key=self.__class__.api_key)
        except Exception as e:
            logging.exception(e)
            logging.info(f"api_key={self.__class__.api_key}")
            raise e

    def get_chat(self, source):
        if source not in self.__chat:
            self.__chat[source] = self.__client.chats.create(
                model=self.__class__.model_name
            )
        return self.__chat[source]

    def do_ask(
        self, query: str, source: Optional[str] = None, fpath: Optional[Path] = None
    ) -> GeminiResponse:
        
        contents = query
        
        if fpath:
            uploaded = self.__client.files.upload(file=fpath)
            contents = [uploaded, query]
            
        if source:
            chat = self.get_chat(source)
            res = chat.send_message(contents)
            return GeminiResponse(content=res.text)

        response = self.__client.models.generate_content(
            model=self.__class__.model_name, contents=contents
        )

        # md = markdown.Markdown()
        # # Now let's test it out:
        # html = md.convert(' '.join(response.text))
        # print(html)
        # print(md.images)

        resp = GeminiResponse(content=response.text)
        return resp

    def do_ask_image(self, img_path: Path, query: str):

        uploaded = self.__client.files.upload(file=img_path)

        response = self.__client.models.generate_content(
            model=self.__class__.model_name, contents=[uploaded, query]
        )
        rich.print(response)

    def do_ask_json(self, query: str, source: Optional[str] = None) -> GeminiResponse:

        if source:
            chat = self.get_chat(source)
            res = chat.send_message(query)
            return GeminiResponse(content=res.text)

        response = self.__client.models.generate_content(
            model=self.__class__.model_name,
            contents=query,
            config={
                "response_mime_type": "application/json",
                # "response_schema": list[Recipe],
            },
        )

        # md = markdown.Markdown()
        # # Now let's test it out:
        # html = md.convert(' '.join(response.text))
        # print(html)
        # print(md.images)

        resp = GeminiResponse(content=response.text)
        return resp
    
    def do_ask_schema(self, query: str, schema: BaseModel):
        response = self.__client.models.generate_content(
            model=self.__class__.model_name,
            contents=query,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema
            )
        )
        return response.parsed
