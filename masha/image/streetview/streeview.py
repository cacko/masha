import logging
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field
from cachable.request import Request
from masha.image.models import GeoLocation
from urllib.parse import urlencode, urlparse, ParseResult
import hashlib
import hmac
import base64
from masha.image.streetview.image import StreetViewImage, PanoramaImage
from masha.image.streetview.s3 import S3
from .panorama import Panorama
from .models import NoImageFound, Status, STREET_ENDPOINT, METADATA_ENDPOINT
import rich.repr
from corestring import string_hash
import re
from rich import print
from masha.image.config import image_config

PATTERN_GPS = re.compile(r"(-?\d+\.\d+)[,\s]+(-?\d+\.\d+)")


class MetadataParams(BaseModel):
    key: str
    location: str

    @property
    def url(self) -> str:
        return f"{METADATA_ENDPOINT}?{urlencode(self.model_dump())}"


class StreetViewParams(BaseModel):
    key: str
    location: str
    size: str = Field(default="640x640")
    fov: int = Field(default=120)
    pitch: int = Field(default=0)
    source: str = Field(default="outdoor")

    @property
    def url(self) -> str:
        return f"{STREET_ENDPOINT}?{urlencode(self.model_dump())}"


class Location(BaseModel):
    lat: float
    lng: float


class MetadataError(BaseModel):
    status: Status


@rich.repr.auto
class MetadataResponse(BaseModel):
    date: str
    location: Location
    pano_id: str
    status: Status

    @classmethod
    def load(cls, **kwds):
        status = Status(kwds.get("status"))
        match status:
            case Status.OK:
                return MetadataResponse(**kwds)
            case Status.ZERO:
                return MetadataError(**kwds)
        raise NotImplementedError


class StreeViewMeta(type):
    __apikey: str = ""
    __apiSecret: str = ""
    __instances: dict[str, "StreetView"] = {}

    def __call__(cls, query: str) -> "StreeViewMeta":
        k = string_hash(query)
        if not k in cls.__instances:
            cls.__instances[k] = type.__call__(
                cls,
                api_key=cls.__apikey,
                api_secret=cls.__apiSecret,
                location=cls.query2location(query),
            )
        return cls.__instances[k]

    def register(cls, api_key: str, api_secret: str):
        cls.__apikey = api_key
        cls.__apiSecret = api_secret

    def query2location(cls, query: str) -> Optional[GeoLocation]:
        try:
            url = f"https://geo.cacko.net/api/address/{query}"
            if m_gps := PATTERN_GPS.search(query):
                lat, lng = m_gps.groups()
                url = f"https://geo.cacko.net/api/gps/{lat}/{lng}"
            geo_res = Request(
                url.replace(
                    image_config.streetview.url, image_config.streetview.local_url
                )
            )
            geo = GeoLocation(**geo_res.json)
            print(geo)
            return geo
        except AssertionError:
            return None


class StreetView(object, metaclass=StreeViewMeta):
    __metadata: Optional[MetadataResponse] = None
    __is_panorama: bool = False

    def __init__(self, api_key: str, api_secret: str, location: GeoLocation) -> None:
        self.__apiKey = api_key
        self.__apiSecret = api_secret
        self.__location = location

    @property
    def location(self) -> GeoLocation:
        return self.__location

    @property
    def is_panorama(self) -> bool:
        return self.__is_panorama

    @property
    def metadata(self) -> Optional[MetadataResponse]:
        try:
            assert self.__metadata
        except AssertionError:
            location = self.__location
            params = MetadataParams(
                key=self.__apiKey,
                location=",".join(map(str, [*location.location])),
            )
            signed_url = self.__sign(params.url)
            rs = Request(signed_url)
            metadata = MetadataResponse.load(**rs.json)
            if isinstance(metadata, MetadataError):
                raise NoImageFound
            self.__metadata = metadata
        return self.__metadata

    @property
    def image(self) -> Optional[Path]:
        try:
            print(self.metadata)
            assert isinstance(self.metadata, MetadataResponse)
            assert self.metadata.pano_id
            pan_image = PanoramaImage(Panorama.get_object(self.metadata.pano_id))
            pan_path = pan_image.path
            assert pan_path
            assert pan_path.exists()
            self.__is_panorama = True
            return pan_path
        except AssertionError:
            location = self.__location
            params = StreetViewParams(
                key=self.__apiKey,
                location=",".join(map(str, [*location.location])),
            )
            signed_url = self.__sign(params.url)
            logging.debug(signed_url)
            assert signed_url
            img_path = StreetViewImage(signed_url).path
            return img_path

    @property
    def s3key(self):
        return S3.src_key(self.image.name)

    def __sign(self, url: str):
        urlparts: ParseResult = urlparse(url)
        to_sign = f"{urlparts.path}?{urlparts.query}".encode()
        decoded_kety = base64.urlsafe_b64decode(self.__apiSecret)
        signature = hmac.new(decoded_kety, to_sign, hashlib.sha1)
        encoded_signature = base64.urlsafe_b64encode(signature.digest())
        return f"{url}&signature={encoded_signature.decode()}"
