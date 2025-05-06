from pathlib import Path
from urllib import response
from google import genai
from google.genai import types


import logging
from typing import Optional
from pydantic import BaseModel
import rich

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

    def ask(cls, query: str) -> GeminiResponse:
        return cls().do_ask(query)
    
    def ask_image(cls, img_path: Path, query: str):
        return cls().do_ask_image(img_path=img_path, query=query)



import markdown
from markdown.treeprocessors import Treeprocessor
from markdown.extensions import Extension

# class ImgExtractor(Treeprocessor):
#     def run(self, doc):
#         "Find all images and append to markdown.images. "
#         self.md.images = []
#         for image in doc.findall('.//img'):
#             self.md.images.append(image.get('src'))

# # Then tell markdown about it

# class ImgExtExtension(Extension):
#     def extendMarkdown(self, md):
#         img_ext = ImgExtractor(md)
#         md.treeprocessors.register(img_ext, 'img_ext', 15)


class Gemini(object, metaclass=GeminiMeta):
    def __init__(self) -> None:
        try:
            self.__client = genai.Client(api_key=self.__class__.api_key)
        except Exception as e:
            logging.exception(e)
            logging.info(f"api_key={self.__class__.api_key}")
            raise e

    def do_ask(self, query: str) -> GeminiResponse:
        response = self.__client.models.generate_content(
            model=self.__class__.model_name, contents=query
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
