import typer
from pathlib import Path
from typing_extensions import Annotated
from coreimage.terminal import print_term_image
from masha.image.cli import cli
from masha.image.router import router
from fastapi import UploadFile, File
from stenogen.shapes import PixelsGenerator
from PIL import Image
from masha.core.request import make_multipart_response, uploaded_file


@cli.command()
def pixel(src_path: Annotated[Path, typer.Argument()]):
    print_term_image(image_path=src_path, height=20)
    img = Image.open(src_path.as_posix())
    res = PixelsGenerator.draw(
        src_path, width=img.width, height=img.height, block_size=100
    )
    assert res
    print_term_image(image_path=res, height=20)


@router.post("/pixel/{block_size}")
async def api_pixel(
    file: Annotated[UploadFile, File()],
    block_size: Annotated[int, Path(title="blopck size")],
):
    tmp_path = await uploaded_file(file)
    try:
        block_size = 100
    except ValueError:
        block_size = 30
    img = Image.open(tmp_path.as_posix())
    res = PixelsGenerator.draw(
        tmp_path, width=img.width, height=img.height, block_size=block_size
    )
    return make_multipart_response(image_path=res)
