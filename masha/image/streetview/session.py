import time
from cachable.request import Request, Method
from urllib.parse import urlencode
from pydantic import BaseModel
from .models import TILE_ENDPOIND
from corestring import clean_newlines
class SessionParams(BaseModel):
    key: str

    @property
    def create_url(self) -> str:
        return f"{TILE_ENDPOIND}/createSession?{urlencode(self.model_dump())}"

    @property
    def data(self) -> dict[str, str]:
        return dict(mapType="streetview", language="en-US", region="UK")


class Session(BaseModel):
    session: str
    expiry: int
    tileHeight: int
    tileWidth: int
    imageFormat: str

    @classmethod
    def create(cls, params: SessionParams) -> "Session":
        rs = Request(params.create_url, method=Method.POST, json=params.data)
        return cls(**rs.json)

    @property
    def is_expired(self) -> bool:
        return time.time() < self.expiry

    @property
    def id(self) -> str:
        
        return clean_newlines(self.session)
