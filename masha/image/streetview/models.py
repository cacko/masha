from enum import StrEnum
from pydantic import BaseModel

STREET_ENDPOINT = "https://maps.googleapis.com/maps/api/streetview"
METADATA_ENDPOINT = "https://maps.googleapis.com/maps/api/streetview/metadata"
TILE_ENDPOIND = "https://tile.googleapis.com/v1"
STATICMAP_ENDPOINT = "https://maps.googleapis.com/maps/api/staticmap"

class Status(StrEnum):
    OK: str = "OK"
    ZERO: str = "ZERO_RESULTS"


class ImageryType(StrEnum):
    OUTDOOR = "outdoor"
    DEFAULT = "default"


class Tile(BaseModel):
    x: int
    y: int
    z: int
    url: str

class NoImageFound(Exception):
    pass