import typer
from pathlib import Path
from typing_extensions import Annotated
from masha.image.cli import cli
from masha.image.diffusers import Diffusers
from masha.image.huggingface import stablediffusion
from masha.image.router import router
from masha.image.config import StyleConfig, image_config, GENERATED_PATH
from fastapi import HTTPException
import logging
from coreimage.terminal import print_term_image
from coreimage.organise.concat import Concat
from masha.image.streetview.s3 import S3
from masha.image.streetview.streeview import StreetView


@cli.command()
def streetview(
    gps: Annotated[list[str], typer.Argument()],
    style: Annotated[
        list[StyleConfig.styles_enum], # type: ignore
        typer.Option("-s", "--style"),
    ] = None,
    output_directory: Annotated[
        Path, typer.Option("-o", "--output_directory")
    ] = GENERATED_PATH,
):
    stablediffusion.is_superuser = True
    results = []
    street_view = StreetView(" ".join(gps))
    outdir = Path(output_directory)
    print_term_image(image_path=street_view.image, height=30)
    for st in style:
        params = image_config.get_style(st)
        cls = Diffusers.cls_for_option(params.model)
        params = cls.pipelineParams(**params.model_dump(exclude=["model", "name"]))
        res = cls.from_img(img_path=street_view.image, params=params)
        assert res
        final_paths = res.save_to(output_dir=outdir)
        for final_path in final_paths:
            results.append(final_path)
            try:
                print_term_image(image_path=final_path, height=30)
            except Exception:
                logging.warning("cannot display image")
            logging.info(res.text)
            logging.info(final_path)

    if len(results) > 3:
        concat_results, _ = Concat(outdir).concat_from_paths(results)
        print_term_image(image_path=concat_results, height=30)


@router.get("/streetview/{style}/{gps}")
async def api_streetview(
    style: Annotated[StyleConfig.styles_enum, Path(title="style")], # type: ignore
    gps: Annotated[str, Path(title="gps")],
):
    try:
        street_view = StreetView(gps)
        raw_url = S3.get_url(street_view.s3key)
        params = image_config.get_style(style)
        cls = Diffusers.cls_for_option(params.model)
        params = cls.pipelineParams(**params.model_dump(exclude=["model", "name"]))
        print(cls)
        res = cls.from_img(
            img_path=street_view.image,
            params=params,
            extra_exif=dict(DocumentName=raw_url),
        )
        assert res
        s3key = S3.upload(src=res.image[0], dst=res.image[0].name)
        return {
            "name": street_view.location.name,
            "raw_url": raw_url,
            "url": S3.get_url(s3key),
        }

    except Exception as e:
        logging.exception(e)
        raise HTTPException(500)
