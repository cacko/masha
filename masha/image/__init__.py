import json
from uuid import uuid4
import requests
from pathlib import Path
from corefile import TempPath
from masha.core.request import make_multipart_response, make_response, uploaded_file
from masha.image.huggingface.sd_types import register_models
from masha.image.image2text import InvoiceReader
from masha.image.variation import Variation
from masha.image.upscale import Upscale
from coreimage.terminal import print_term_image
from masha.image.diffusers import Diffusers
from coreimage.find import find_images
from masha.image.console import cmd_upload
from masha.image.qrcode import QRCode, register_qrcodes
from .prompt import register_prompts

from .models import (
    Resolutions,
    UploadRequest,
    Category,
    VariationPipelineParams,
)
import logging
from masha.image.config import image_config
from urllib.request import urlretrieve
import filetype
from .huggingface.scripts import (
    convert_vae as script_convert_vae,
    convert_ckpt as script_convert_ckpt,
    convert_sdxl as script_convert_sdxl,
    convert_controlnet as script_convert_controlnet,
)
from .cli import cli
from .router import router


from fastapi import Request, HTTPException, File, UploadFile, Form
import typer
from typing_extensions import Annotated


Variation.register(image_config.variation.model)
Upscale.register(image_config.upscale.root)


# P2PStandard.register(app_config.image.pix2pix_model)

register_models()
register_prompts()
register_qrcodes()

from masha.image.commands import *

OPTIONS = dict(
    model=Diffusers.options,
    resolution=Resolutions.keys(),
    category=Category.values(),
    template=Diffusers.templates,
    qrcode_templates=QRCode.templates,
    qrcode_models=QRCode.models,
    styles=image_config.style_options
)

@cli.command()
def variation(
    src_path: Annotated[Path, typer.Argument()],
    images: Annotated[int, typer.Option("-n")] = 1,
    scale: Annotated[int, typer.Option("-s")] = 3,
    out_path: Annotated[Path, typer.Option("-o")] = Path("."),
):
    if out_path.is_dir():
        out_path = out_path / f"{src_path.stem}_variations{src_path.suffix}"
    logging.debug(out_path)
    print_term_image(image_path=src_path, height=20)
    res = Variation.variation(
        src_path=src_path,
        out_path=out_path,
        params=VariationPipelineParams(
            guidance_scale=scale, num_images_per_prompt=images
        ),
    )
    assert res
    logging.debug(res)
    print_term_image(image_path=res, height=20)


@router.post("/variation/{id}")
async def api_variation(
    file: Annotated[UploadFile, File()],
    id: Annotated[str, Path(title="blopck size")],
    data: Annotated[str, Form()],
):
    try:
        tmp_path = await uploaded_file(file)
        params = json.loads(data)
        out_path = TempPath(f"{tmp_path.stem}_variations_{tmp_path.suffix}")
        res_path = Variation.variation(
            src_path=tmp_path,
            out_path=out_path,
            params=VariationPipelineParams(**params),
        )
        assert res_path
        return make_multipart_response(res_path)
    except Exception as e:
        logging.exception(e)
        raise HTTPException(500)


# @bp.route("/tpl/<template>/<prompt>", methods=["GET"])
# def get_tpl_image(template: str, prompt: str):
#     try:
#         tpl = image_config.get_template(template)
#         input_params = dict(prompt=f"£{prompt}£")
#         assert tpl
#         model = tpl.models[0]
#         input_params = tpl.apply(input_params=input_params)
#         assert model
#         cls = Diffusers.cls_for_option(model)
#         params = cls.pipelineParams(**input_params)
#         image_result = cls.from_text(params=params)
#         assert image_result
#         cls.release()
#         return send_file(image_result.image[0].as_posix())
#     except (AssertionError, LookupError, TypeError):
#         abort(404)


# def gps_to_location(lat, lon) -> str:
#     logging.debug([lat, lon])
#     m = mgrs.MGRS()
#     nd, nm, ns = m.ddtodms(lat)
#     ed, em, es = m.ddtodms(lon)
#     dr = "E" if lon > 0 else "W"
#     qr = f"{nd:.0f}° {nm:.0f}' {ns:.0f}\"N, {abs(ed):.0f}° {abs(em):.0f}'{abs(es):.0f}\"{dr}"
#     logging.debug(qr)
#     res = Wolfram.query(qr)
#     assert res.message
#     rows = res.message.split("\n")
#     assert len(rows) > 2
#     result = [rows[2].split("(")[0]]
#     for row in rows:
#         if row.strip().startswith("notable building"):
#             result.append(row.split("|")[1])
#         if row.strip().startswith("notable park"):
#             result.append(row.split("|")[1])
#     return ",".join(result)


@cli.command()
def upscale(
    path: Annotated[Path, typer.Argument()],
    scale: Annotated[int, typer.Option("-s", "--scale")] = 4,
    output: Annotated[Path, typer.Option("-o", "--output")] = None,
):
    def get_dst(src_path: Path) -> Path:
        if not output:
            return src_path.parent / f"{src_path.stem}-upscaled{src_path.suffix}"
        if output.is_dir():
            return output / f"{src_path.stem}-upscaled{src_path.suffix}"
        return output

    src_paths = [path]
    logging.info(src_paths)
    for src in find_images(src_paths):
        src_path = Path(src)
        dest = get_dst(src_path=src_path)
        res = Upscale.upscale(src_path=src_path, dst_path=dest, scale=scale)
        assert res
        logging.info(res)
        print_term_image(image_path=res, height=30)


@router.post("/upscale")
async def api_upscale(
    file: Annotated[UploadFile, File()],
):
    try:
        tmp_path = await uploaded_file(file)
        res_path = Upscale.upscale(src_path=tmp_path)
        assert res_path
        return make_multipart_response(res_path)
    except AssertionError:
        raise HTTPException(404)


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
        kind = filetype.guess(tmp_path.as_posix())
        mime = kind.mime
        fp = tmp_path.open("rb")
        assert kind
        params = dict(
            files=dict(
                file=(f"{tmp_path.stem}.{kind.extension}", fp, mime, {"Expires": "0"})
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


@cli.command()
def upload(
    path: Annotated[list[str], typer.Argument()],
    category: Annotated[Category, typer.Option("-c", "--category")],
    upscale: Annotated[bool, typer.Option("-u", "--upscale")] = False,
):
    cmd_upload(path, category, upscale)


@router.get("/options")
def api_options():
    try:
        return OPTIONS
    except AssertionError as e:
        logging.exception(e)
        raise HTTPException(404)
    except Exception as e:
        logging.exception(e)
        raise HTTPException(500)


@cli.command()
def convert_ckpt(
    model_path: Annotated[Path, typer.Argument()],
    out_path: Annotated[Path, typer.Option("-o")],
    safe_tensors: Annotated[bool, typer.Option("-s")] = False,
    is_version1: Annotated[bool, typer.Option("-v1")] = False,
):
    script_convert_ckpt(
        in_model=model_path,
        dump_to=out_path,
        safe_tensors=safe_tensors,
        v1=is_version1,
    )


@cli.command()
def convert_sdxl(
    model_path: Annotated[Path, typer.Argument()],
    out_path: Annotated[Path, typer.Option("-o")],
):
    script_convert_sdxl(
        in_model=model_path,
        dump_to=out_path,
    )


@cli.command()
def convert_vae(
    model_path: Annotated[Path, typer.Argument()],
    out_path: Annotated[Path, typer.Option("-o")],
):
    script_convert_vae(
        in_model=model_path,
        dump_to=out_path,
    )


@cli.command()
def cli_convert_controlnet(
    model_path: Annotated[Path, typer.Argument()],
    out_path: Annotated[Path, typer.Option("-o")],
    safe_tensors: Annotated[bool, typer.Option("-s")] = False,
):
    script_convert_controlnet(
        in_model=model_path, dump_to=out_path, safe_tensors=safe_tensors
    )


@cli.command()
def image2text(
    path: Annotated[Path, typer.Argument()],
):

    src_paths = [path]
    logging.info(src_paths)
    
    for src in find_images(src_paths):
        src_path = Path(src)
        print_term_image(image_path=src_path)
        woker = InvoiceReader()
        res = woker.getAnswer(src_path)
        assert res
        logging.info(res)




@router.post("/img2text")
async def api_img2text(
    file: Annotated[UploadFile, File()]
):
    try:
        tmp_path = await uploaded_file(file)
        res = InvoiceReader().getAnswer(tmp_path)
        assert res
        return make_response(message=res)
    except AssertionError:
        raise HTTPException(404)