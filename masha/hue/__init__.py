from enum import Enum, IntEnum
import logging
from lambo.hue.client import Hue
from masha.code.summarization import CodeSummarization
import click
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException
import typer
from typing_extensions import Annotated
from masha.config import app_config
from pydantic import BaseModel

router = APIRouter()
cli = typer.Typer()


class ContentLight(BaseModel):
    duration: int
    colors: list[str]


class ALERT_COLOR(Enum):
    SOUNDCODE = ["1BB94B", "34B0EE"]
    PAKI_RABAN = ["FF8CFF", "FFFF46"]


class ALERT_DURATION(IntEnum):
    SOUNDCODE = 1000
    PAKI_RABAN = 5000


Hue.register(hostname=app_config.hue.hostname, username=app_config.hue.username)
# Hue.signaling(duration=1000, colors=["DDDD00", "DD1FD0"])


@cli.command()
def alert(alert: Annotated[str, typer.Argument()]):
    try:
        colors = ALERT_COLOR[alert.upper()]
        duration = ALERT_DURATION[alert.upper()]
        alert_content = ContentLight(
            duration=duration,
            colors=colors.value,
        )
        Hue.signaling(**alert_content.model_dump())
    except Exception as e:
        logging.exception(e)


# @router.put("/diff")
# async def route_tag(request: Request):
#     diff = await request.body()
#     try:
#         msg = CodeSummarization.summarization(diff.decode())
#         return {"message": msg}
#     except AttributeError:
#         pass
#     raise HTTPException(404)
