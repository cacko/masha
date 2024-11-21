from pathlib import Path
from masha.core.request import make_multipart_response
from masha.image.huggingface.stablediffusion import StableDiffusion
from coreimage.terminal import print_term_image
from masha.image.diffusers import Diffusers
from coreimage.organise.concat import Concat
from masha.image.cli import cli
from masha.image.router import router
from masha.image.huggingface.utils import txt2img_iterations
import logging
from masha.image.config import TemplateConfig, image_config, GENERATED_PATH
from fastapi import HTTPException, Request
import typer
from typing_extensions import Annotated


@cli.command()
def txt2img(
    prompt: Annotated[list[str], typer.Argument()],
    models: Annotated[
        list[Diffusers.options_enum], typer.Option("-m", "--models")
    ] = None,
    output_directory: Annotated[
        Path, typer.Option("-o", "--output_directory")
    ] = GENERATED_PATH,
    guidance_scale: Annotated[float, typer.Option("-g", "--guidance_scale")] = None,
    width: Annotated[int, typer.Option("-w", "--width")] = None,
    height: Annotated[int, typer.Option("-h", "--height")] = None,
    num_inference_steps: Annotated[
        int, typer.Option("-i", "--num_inference_steps")
    ] = None,
    upscale: Annotated[bool, typer.Option("-u", "--upscale")] = False,
    auto_prompt: Annotated[str, typer.Option("-a", "--auto_prompt")] = None,
    negative_prompt: Annotated[str, typer.Option("-n", "--negative_prompt")] = None,
    seed: Annotated[int, typer.Option("-s", "--seed")] = None,
    strength: Annotated[float, typer.Option("-st", "--strength")] = None,
    clip_skip: Annotated[int, typer.Option("-cs", "--clip_skip")] = None,
    aspect_ratio: Annotated[
        Diffusers.resolutions_enum,
        typer.Option("-r", "--aspect_ratio"),
    ] = None,
    category: Annotated[
        Diffusers.categories_enum, typer.Option("-ct", "--category")
    ] = None,
    editing_prompt: Annotated[str, typer.Option("-e", "--editing-prompt")] = None,
    template: Annotated[
        Diffusers.templates_enum, typer.Option("-t", "--template")
    ] = None,
    all_prompts: Annotated[bool, typer.Option("--all-prompts")] = False,
    all_templates: Annotated[bool, typer.Option("--all-templates")] = False,
    template_category: Annotated[
        TemplateConfig.categories_enum,  # type: ignore
        typer.Option("-tc", "--template-category"),
    ] = None,
):
    StableDiffusion.is_superuser = True
    outdir = output_directory
    results = []
    inputParams = dict(
        prompt=" ".join(prompt),
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        negative_prompt=negative_prompt,
        seed=seed,
        upscale=upscale,
        aspect_ratio=aspect_ratio,
        editing_prompt=editing_prompt,
        clip_skip=clip_skip,
        is_super_user=True,
        strength=strength,
    )
    for cls, params in txt2img_iterations(
        inputParams=inputParams,
        models=models,
        category=category,
        template=template,
        template_category=template_category,
        auto_prompt=auto_prompt,
        all_prompts=all_prompts,
        all_templates=all_templates,
    ):
        res = cls.generate_from_text(params=params)
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


@router.post("/txt2img/{prompt}")
async def txt2img(request: Request, prompt: Annotated[str, Path(title="prompt")]):
    try:
        form_data = await request.json()
        assert form_data
        input_params = form_data
        model = form_data.get("model")
        try:
            template = input_params.get("template")
            assert template
            del input_params["template"]
            tpl = image_config.get_template(template)
            assert tpl
            if not model:
                model = tpl.models[0]
            input_params = tpl.apply(input_params=input_params)
        except AssertionError:
            pass
        assert model
        instance = Diffusers.cls_for_option(model)()
        params = instance.pipelineParams(**input_params)
        image_result = instance.generate_from_text(params=params)
        assert image_result
        return make_multipart_response(
            image_path=image_result.image[0], message=image_result.text
        )
    except Exception as e:
        logging.exception(e)
        raise HTTPException(500)
