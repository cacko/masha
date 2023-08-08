from pathlib import Path
import logging
from coreimage.terminal import print_term_image
from coreimage.transform.upscale import Upscale
from corefile import filepath
import filetype
import requests
from pydantic import BaseModel


class JSONData(BaseModel):
    category: str
    botyo_id: str


def cmd_upload(path: list[str], category: str, upscale: bool):
    def get_paths(pths: list[str]):
        for pth in pths:
            yield from filepath(Path(pth).absolute(), suffixes=[".jpg", ".png"])

    for pth in get_paths(path):
        p = Path(pth)
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
        kind = filetype.guess(p.as_posix())
        assert kind
        mime = kind.mime
        with p.open("rb") as fp:
            params = {
                "files": {
                    "file": (
                        f"{p.stat}.{kind.extension}",
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
