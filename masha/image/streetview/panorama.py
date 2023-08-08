from itertools import product, takewhile
from typing import Generator, Optional
from cachable.request import Request
from urllib.parse import urlencode
import rich.repr
from pydantic import BaseModel
from .models import TILE_ENDPOIND, Tile
from .session import Session, SessionParams
from rich import print

class TileParams(BaseModel):
    session: str
    key: str
    panoId: str

    def url(self, z: int, x: int, y: int) -> str:
        params = urlencode(self.model_dump())
        return f"{TILE_ENDPOIND}/streetview/tiles/{z}/{x}/{y}?{params}"


class MetadataParams(BaseModel):
    session: str
    key: str
    panoId: str

    @property
    def url(self) -> str:
        return f"{TILE_ENDPOIND}/streetview/metadata?{urlencode(self.model_dump())}"


@rich.repr.auto
class Metadata(BaseModel):
    panoId: str
    lat: float
    lng: float
    imageHeight: int
    imageWidth: int
    tileHeight: int
    tileWidth: int
    heading: float
    tilt: float
    roll: float

    @classmethod
    def create(cls, params: MetadataParams) -> "Metadata":
        rs = Request(params.url)
        return cls(**rs.json)

    @property
    def n_cols(self) -> int:
        return self.imageWidth // self.tileWidth

    @property
    def n_rows(self) -> int:
        return self.imageHeight // self.tileHeight

    @property
    def zoom(self) -> int:
        return list(takewhile(
            lambda x: 2**x * self.tileWidth >= self.imageWidth, range(5, 0, -1)
        )).pop()


class PanoramaMeta(type):
    __apiKey: Optional[str] = None

    def __call__(cls, pano_id: str):
        return type.__call__(cls, cls.__apiKey, pano_id)

    def register(cls, api_key: str):
        cls.__apiKey = api_key

    def get_object(cls, pano_id: str) -> "Panorama":
        return cls(pano_id=pano_id)


class Panorama(metaclass=PanoramaMeta):
    __session: Optional[Session] = None
    __metadata: Optional[Metadata] = None

    def __init__(self, api_key: str, pano_id: str) -> None:
        self.__api_key = api_key
        self.__pano_id = pano_id

    @property
    def session(self) -> str:
        if not self.__session:
            self.__session = Session.create(SessionParams(key=self.__api_key))
        if self.__session.is_expired:
            self.__session = Session.create(SessionParams(key=self.__api_key))
        return self.__session
    
    @property
    def id(self) -> str:
        return self.__pano_id

    @property
    def metadata(self) -> Metadata:
        try:
            assert self.__metadata
            return self.__metadata
        except AssertionError:
            self.__metadata = Metadata.create(
                MetadataParams(
                    session=self.session.id, key=self.__api_key, panoId=self.__pano_id
                )
            )
        return self.__metadata

    def tiles(self, session: Optional[Session] = None) -> Generator[Tile, None, None]:
        params = TileParams(
            session=self.session.id, key=self.__api_key, panoId=self.__pano_id
        )
        z = self.metadata.zoom
        for x, y in product(
            list(range(self.metadata.n_cols)), list(range(self.metadata.n_rows))
        ):
            yield Tile(url=params.url(z, x, y), x=x, y=y, z=z)
