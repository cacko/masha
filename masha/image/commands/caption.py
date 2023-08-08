import typer
from pathlib import Path
from typing_extensions import Annotated
from masha.image.cli import cli
from masha.image.router import router
from rich import print
from fastapi import UploadFile, File
from masha.core.request import uploaded_file
from coreimage.terminal import print_term_image
from masha.image.caption import ImageCaption

@router.post("/img2caption")
async def api_img2caption(
    file: Annotated[UploadFile, File()],
):
    tmp_path = await uploaded_file(file)
    res = ImageCaption.caption(
        image=tmp_path
    )
    return {"response": res}


@cli.command()
def img2caption(img_path: Annotated[Path, typer.Argument()]):
    print_term_image(image_path=img_path, height=20)
    res = ImageCaption.caption(
        image=img_path
    )
    print(res)