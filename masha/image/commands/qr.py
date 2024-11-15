import typer
from pathlib import Path
from typing_extensions import Annotated
from coreimage.terminal import print_term_image
from masha.image.cli import cli
from masha.image.qrcode import QRCode
from masha.image.router import router
from masha.core.request import make_multipart_response, uploaded_file
from masha.image.config import GENERATED_PATH
import shutil
import logging
from fastapi import Request, HTTPException
from corestring import split_with_quotes



@cli.command()
def txt2qr(
    qrcode: Annotated[list[str], typer.Argument()],
    output_directory: Annotated[
        Path, typer.Option("-o", "--output_directory")
    ] = GENERATED_PATH,
):
    assert qrcode
    result = QRCode.txt2qr(qrcode)
    assert result.exists()
    print_term_image(image_path=result, height=40)
    dst = Path(output_directory) / result.name
    shutil.copy(result.as_posix(), dst.as_posix())


@router.post("/txt2qr/{prompt}")
async def api_txt2qr(request: Request, prompt: Annotated[str, Path(title="prompt")]):
    try:
        form_data = await request.json()
        logging.debug(form_data)
        assert form_data
        code = form_data.get("code")
        logging.debug(code)
        image_result = QRCode.txt2qr(code)
        assert image_result
        return make_multipart_response(image_path=image_result)
    except Exception as e:
        logging.exception(e)
        raise HTTPException(500, detail=f"{e}")


@cli.command()
def qr2img(
    qrcode: Annotated[list[str], typer.Argument()],
    template: Annotated[QRCode.templates_enum, typer.Option("-t", "--template")] = None,
    model: Annotated[QRCode.models_enum, typer.Option("-m", "--model")] = None,
    output_directory: Annotated[
        Path, typer.Option("-o", "--output_directory")
    ] = GENERATED_PATH,
    controlnet_conditioning_scale: Annotated[
        float, typer.Option("-c", "--controlnet_conditioning_scale")
    ] = None,
    guidance_scale: Annotated[float, typer.Option("-g", "--guidance_scale")] = None,
    prompt: Annotated[str, typer.Option("-p", "--prompt")] = None,
    auto_prompt: Annotated[str, typer.Option("-a", "--auto_prompt")] = None,
    seed: Annotated[int, typer.Option("-s", "--seed")] = None,
    num_inference_steps: Annotated[
        int, typer.Option("-i", "--num_inference_steps")
    ] = None,
    image: Annotated[Path, typer.Option("--image")] = None,
    strength: Annotated[float, typer.Option("-st", "--strength")] = None,
):
    assert qrcode
    qrimage = QRCode.get_qrcode_image(qrcode)
    assert qrimage
    qr_cls = QRCode.get_template(template)
    if image and image.exists():
        print_term_image(image_path=image, height=20)
    result = qr_cls.code2img(
        model=model,
        qr=qrimage,
        prompt=prompt,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        guidance_scale=guidance_scale,
        auto_prompt=auto_prompt,
        seed=seed,
        num_inference_steps=num_inference_steps,
        image=image,
        strength=strength,
    )
    assert result
    print_term_image(image_path=result, height=40)
    dst = Path(output_directory) / result.name
    shutil.copy(result.as_posix(), dst.as_posix())


@router.post("/qr2img/{prompt}")
async def api_qr2img(request: Request, prompt: Annotated[str, Path(title="prompt")]):
    try:
        form_data = await request.json()
        assert form_data
        input_params = form_data
        logging.warning(input_params)
        code = form_data.get("code")
        assert code
        template = form_data.get("template", "default")
        logging.debug(code)
        del input_params["template"]
        del input_params["code"]
        qrimage = QRCode.get_qrcode_image(split_with_quotes(code))
        assert qrimage
        cls = QRCode.get_template(template)
        image_result = cls.code2img(qr=qrimage, **input_params)
        assert image_result
        return make_multipart_response(image_path=image_result)
    except Exception as e:
        logging.exception(e)
        raise HTTPException(500)
