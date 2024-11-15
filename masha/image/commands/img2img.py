import json
from pathlib import Path
from masha.core.request import make_multipart_response, uploaded_file
from masha.image.caption import ImageCaption
from masha.image.huggingface.stablediffusion import StableDiffusion
from coreimage.terminal import print_term_image
from masha.image.diffusers import  Diffusers
from coreimage.organise.concat import Concat
from masha.image.huggingface.utils import img2img_iterations
import logging
from masha.image.config import StyleConfig, TemplateConfig, GENERATED_PATH
from masha.image.cli import cli
from masha.image.router import router

from fastapi import File, HTTPException, UploadFile, Form
import typer
from typing_extensions import Annotated


@cli.command()
def img2img(
    img_path: Annotated[Path, typer.Argument()],
    models: Annotated[list[Diffusers.options_enum], typer.Option("-m", "--models")] = [
        None
    ],
    output_directory: Annotated[
        Path, typer.Option("-o", "--output_directory")
    ] = GENERATED_PATH,
    guidance_scale: Annotated[float, typer.Option("-g", "--guidance_scale")] = None,
    strength: Annotated[float, typer.Option("-st", "--strength")] = None,
    num_images_per_prompt: Annotated[
        int, typer.Option("-c", "--num_images_per_prompt")
    ] = 1,
    num_inference_steps: Annotated[
        int, typer.Option("-i", "--num_inference_steps")
    ] = None,
    upscale: Annotated[int, typer.Option("-u", "--upscale")] = 2,
    negative_prompt: Annotated[str, typer.Option("-n", "--negative_prompt")] = None,
    category: Annotated[
        Diffusers.categories_enum, typer.Option("-ct", "--categories")
    ] = None,
    style: Annotated[StyleConfig.styles_enum, typer.Option("-s", "--style")] = None, # type: ignore
    all_styles: Annotated[bool, typer.Option("--all-styles")] = False,
    prompt: Annotated[str, typer.Option("-p", "--prompt")] = "",
    generate_caption: Annotated[
        bool, typer.Option("-gc", "--generate-caption")
    ] = False,
    no_auto_caption: Annotated[bool, typer.Option("-nc", "--no-auto-caption")] = False,
    template: Annotated[
        Diffusers.templates_enum, typer.Option("-t", "--template")
    ] = None,
    all_templates: Annotated[bool, typer.Option("--all-templates")] = False,
    template_category: Annotated[
        TemplateConfig.categories_enum, # type: ignore
        typer.Option("-tc", "--template_category"),
    ] = None,
):
    StableDiffusion.is_superuser = True
    outdir = Path(output_directory)
    results = []
    # try:
    #     assert not no_auto_caption
    #     assert any([generate_caption, not prompt])
    #     prompt += f",{ImageCaption.caption(img_path)}"
    # except AssertionError as e:
    #     pass
    inputParams = dict(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        negative_prompt=negative_prompt,
        upscale=upscale,
        num_images_per_prompt=num_images_per_prompt,
        is_super_user=True,
        guidance_scale=guidance_scale,
        strength=strength,
        height=1024,
        width=1024,
    )
    for cls, params in img2img_iterations(
        inputParams=inputParams,
        models=models,
        category=category,
        style=style,
        all_styles=all_styles,
        template=template,
        all_templates=all_templates,
        template_category=template_category,
    ):
        res = cls.from_img(img_path=img_path.resolve(), params=params)
        assert res
        final_paths = res.save_to(output_dir=outdir)
        for final_path in final_paths:
            results.append(final_path)
            try:
                print_term_image(image_path=final_path, height=40)
            except Exception:
                logging.warning("cannot display image")
            logging.info(res.text)
            logging.info(final_path)

    if len(results) > 3:
        concat_results, _ = Concat(outdir).concat_from_paths(results)
        print_term_image(image_path=concat_results, height=40)


@router.post("/img2img/{prompt}")
async def api_img2img(
    file: Annotated[UploadFile, File()],
    prompt: Annotated[str, Path(title="blopck size")],
    data: Annotated[str, Form()],
):
    try:
        tmp_path = await uploaded_file(file)
        form_data = json.loads(data)
        assert form_data
        model = form_data.get("model")
        style = form_data.get("style")
        template = form_data.get("template")
        no_auto_caption = form_data.get("no_auto_caption")
        input_params = form_data
        try:
            prompt = input_params.get("prompt", "")
            assert any([not prompt, not no_auto_caption])
            prompt += f",{ImageCaption.caption(tmp_path)}"
            input_params["prompt"] = prompt
        except AssertionError:
            pass
        for cls, params in img2img_iterations(
            inputParams=input_params,
            models=[model],
            category=None,
            style=style,
            all_styles=False,
            template=template,
            all_templates=False,
            template_category=None,
        ):
            image_result = cls.from_img(tmp_path, params=params)
            assert image_result
            logging.info(image_result.text)
            return make_multipart_response(
                image_path=image_result.image[0], message=image_result.text
            )
    except Exception as e:
        logging.exception(e)
        raise HTTPException(500)
