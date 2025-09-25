from openai import OpenAI
from openai.types.responses import Response
from masha.text.config import OpenAIConfig
from typing import Optional

class ChatGPTMeta(type):
    __instance: Optional["ChatGPT"] = None

    def __call__(cls, *args, **kwds):
        if not cls.__instance:
            cls.__instance = None
        cls.__instance = type.__call__(cls, *args, **kwds)
        return cls.__instance

    def register(cls, cfg: OpenAIConfig):
        cls.__cfg = cfg

    @property
    def api_key(cls) -> str:
        return cls.__cfg.api_key

    @property
    def model_name(cls) -> str:
        return cls.__cfg.model
    
    def ask(cls, query: str, source: Optional[str] = None) -> Response:
        return cls().do_ask(query=query, source=source)

    

class ChatGPT(object, metaclass=ChatGPTMeta):

    def __init__(self):
        self.__client = OpenAI(
            api_key=self.__class__.api_key
        )
        
    def do_ask(self, query: str, **kwds):
        response = self.__client.responses.create(
            model=self.__class__.model_name,
            input=query
        )
        print(response.output_text)
        return response