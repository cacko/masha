from pathlib import Path
from coreimage.terminal import print_term_image
import logging
from masha.image.cli import cli
from masha.image.router import router
from masha.image.config import image_config
from masha.image.console import cmd_upload
import requests
from masha.image.models import (
    UploadRequest,
    Category,
)
import typer
from typing_extensions import Annotated
from fastapi import Request, HTTPException
from corefile import TempPath
from urllib.request import urlretrieve
from corefile import find_mime_extension
from masha.image.upscale import Upscale
from typing import Optional
from masha.text.genai import Gemini
from pydantic import BaseModel
from random import choice


@router.post("/upload2wallies/{image}")
async def api_upload2wallies(
    request: Request, image: Annotated[str, Path(title="prompt")]
):
    try:
        data = await request.json()
        assert data
        upload_params = UploadRequest(**data)
        image_path = Path(image)
        res = requests.get(f"http://192.168.0.10:43211/api/artwork/{image_path.stem}")
        if res.status_code == 200:
            raise AssertionError
        tmp_path = TempPath(f"{uuid4().hex}{image_path.suffix}")
        urlretrieve(upload_params.image_url, tmp_path.as_posix())
        if upload_params.upscale:
            tmp_path = Upscale.upscale(tmp_path)
        mime, extension = find_mime_extension(tmp_path.as_posix())
        fp = tmp_path.open("rb")
        assert mime
        params = dict(
            files=dict(
                file=(f"{tmp_path.stem}.{extension}", fp, mime, {"Expires": "0"})
            ),
            data=dict(category=upload_params.category, botyo_id=image_path.stem),
        )
        logging.info(data)
        res = requests.post(
            "http://192.168.0.10:43211/api/artworks", **params  # type: ignore
        )
        return res.json()
    except AssertionError:
        raise HTTPException(404)


class FluxPrompt(BaseModel):
    prompt: str


@cli.command()
def upload(
    path: Annotated[list[str], typer.Argument()],
    category: Annotated[Optional[Category], typer.Option("-c", "--category")] = None,
    upscale: Annotated[bool, typer.Option("-u", "--upscale")] = False,
):
    cmd_upload(path=path, category=category, upscale=upscale)


@cli.command()
def wallies_prompt(
    category: Annotated[Optional[Category], typer.Option("-c", "--category")] = None,
):

    style = choice(image_config.art.style)
    technique = choice(image_config.art.technique)
    wtype = choice(image_config.art.type)
    form = choice(image_config.art.form)
    palette = choice(image_config.art.palette)
    if not category:
        category = Category.random()

    prompt = (
        f"Generate a single, highly descriptive, and imaginative text-to-image prompt."
        f"The prompt should be for a 'Flux' AI model {category.value} image using {style} style {wtype},"
        f"unexpected subject matter, and {palette} color palettes. Ensure the content is "
        f"unpredictable and wildly creative, suitable for a random generation."
    )
    logging.info(f">> INPUT: {prompt}")
    resp: FluxPrompt = Gemini.ask_schema(query=prompt, schema=FluxPrompt)
    logging.info(f">> RESULT: {resp.prompt}")
    print(resp)
