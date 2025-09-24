from pathlib import Path
import logging
from typing import Optional
from coreimage.terminal import print_term_image
from coreimage.transform.upscale import Upscale
from corefile import filepath
from corefile import find_mime_extension
import requests
from pydantic import BaseModel

from masha.image.models import Category


class JSONData(BaseModel):
    category: str
    botyo_id: str


def cmd_upload(path: list[str], upscale: bool, category: Optional[str] = None):
    def get_paths(pths: list[str]):
        for pth in pths:
            yield from filepath(Path(pth).absolute(), suffixes=[".jpg", ".png"])

    for pth in get_paths(path):
        p = Path(pth)
        try:
            assert category
        except AssertionError:
            category = Category.guess(p.stem)
        assert category    
        print_term_image(image_path=p, height=20)
        logging.info(f"Uploading as {category}")
        if upscale:
            try:
                logging.info("Upscaling....")
                res = Upscale.upscale(src_path=p, scale=2)
                assert res
                p = res
            except AssertionError as e:
                logging.exception(e)
                pass
        mime, extension = find_mime_extension(p.as_posix())
        assert mime
        with p.open("rb") as fp:
            params = {
                "files": {
                    "file": (
                        f"{p.stat}.{extension}",
                        fp,
                        mime,
                        {
                            "Expires": "0"
                        }
                    )
                },
                "data": JSONData(
                    category=category,
                    botyo_id=p.stem
                ).model_dump()
            }
            logging.info(params)
            response = requests.post(
                "http://192.168.0.10:43211/api/artworks",
                **params  # type: ignore
            )
            logging.info(f"Result: {response.status_code}")
