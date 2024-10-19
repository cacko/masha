import typer
from pathlib import Path
from typing_extensions import Annotated
from masha.image.cli import cli
from masha.image.router import router
import logging
import torch
from coreimage.terminal import print_term_image
from masha.image.config import TemplateConfig, GENERATED_PATH
from masha.image.huggingface.stablediffusion import StableDiffusion
from masha.image.huggingface.utils import txt2img_iterations
from masha.image.diffusers import IMAGE_FORMAT, Diffusers
from masha.image.models import IncompatibleTemplate, OutputParams, PipelineParams
from os import environ
from fastapi import File, HTTPException, UploadFile, Form
from masha.image.config import image_config
from coreimage.organise.concat import Concat
from masha.core.request import make_multipart_response, uploaded_file
import json
from masha.image.faceid import FaceId


def get_output_params(params: PipelineParams, device: str = "mps") -> OutputParams:
    prompt = params.output_prompt
    negative_prompt = params.negative_prompt
    generator = torch.Generator(device).manual_seed(-1)
    return OutputParams(
        prompt=prompt,
        guidance_scale=params.guidance_scale,
        num_inference_steps=params.num_inference_steps,
        negative_prompt=negative_prompt,
        generator=generator,
        width=params.output_width,
        height=params.output_height,
        num_images_per_prompt=params.num_images_per_prompt,
        cross_attention_kwargs=params.cross_attention_kwargs,
        clip_skip=params.clip_skip,
        strength=params.strength,
    )


def get_data_root() -> Path:
    return Path(environ.get("HUGGINGROOT", ".")) / "txt2img"


def get_model_path(model_path="IP-Adapter-FaceID/ip-adapter-faceid_sdxl.bin") -> Path:
    return get_data_root() / model_path


def get_lora_path() -> Path:
    return get_data_root() / image_config.lora.root_xl


def get_textinversion_root() -> Path:
    return get_data_root() / image_config.textual_inversion.root_xl


@cli.command()
def img2face(
    img_path: Annotated[Path, typer.Argument()],
    output_name: Annotated[str, typer.Option("-o")] = None,
):
    if not output_name:
        output_name = img_path.stem
    output_path = image_config.face2img.face_store / f"{output_name}.face"
    if faceid := FaceId.create(input=img_path, output=output_path):
        logging.info(
            f"{img_path} saved to {output_path}, age: {faceid.age}, gender: {faceid.race}, gender: {faceid.sex}"
        )


@cli.command()
def face2img(
    faces: Annotated[list[str], typer.Argument()],
    prompt: Annotated[list[str], typer.Option("-p", "--prompt")] = [""],
    models: Annotated[
        list[Diffusers.options_enum], typer.Option("-m", "--models")
    ] = None,
    width: Annotated[int, typer.Option("-w", "--width")] = None,
    height: Annotated[int, typer.Option("-h", "--height")] = None,
    aspect_ratio: Annotated[
        Diffusers.resolutions_enum,
        typer.Option("-r", "--aspect-ratio"),
    ] = None,
    guidance_scale: Annotated[float, typer.Option("-g", "--guidance-scale")] = None,
    num_inference_steps: Annotated[
        int, typer.Option("-i", "--num-inference-steps")
    ] = None,
    upscale: Annotated[bool, typer.Option("-u", "--upscale")] = False,
    clip_skip: Annotated[int, typer.Option("-cs", "--clip-skip")] = None,
    negative_prompt: Annotated[str, typer.Option("-n", "--negative-prompt")] = None,
    num_images_per_prompt: Annotated[
        int, typer.Option("-c", "--num-images-per-prompt")
    ] = 1,
    category: Annotated[
        Diffusers.categories_enum, typer.Option("-ct", "--categories")
    ] = None,
    scale: Annotated[float, typer.Option("-sc", "--scale")] = None,
    template: Annotated[
        Diffusers.templates_enum, typer.Option("-t", "--template")
    ] = None,
    all_templates: Annotated[bool, typer.Option("--all-templates")] = False,
    template_category: Annotated[
        TemplateConfig.categories_enum,  # type: ignore
        typer.Option("-tc", "--template-category"),
    ] = None,
    seed: Annotated[int, typer.Option("-s", "--seed")] = None,
    no_person: Annotated[bool, typer.Option("--no-person")] = None,
    no_race: Annotated[bool, typer.Option("--no-race")] = False,
    no_age: Annotated[bool, typer.Option("--no-age")] = False,
    output_directory: Annotated[
        Path, typer.Option("-o", "--output-directory")
    ] = GENERATED_PATH,
    pag_scale: Annotated[float, typer.Option("-ps", "--pag_scale")] = None,
):
    StableDiffusion.is_superuser = True
    results = []
    outdir = output_directory

    def render(img_pth):
        faceid = FaceId.try_by_name(img_pth)
        print_term_image(image_path=faceid.path_crop)
        inputParams = dict(
            prompt=" ".join(prompt),
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            negative_prompt=negative_prompt,
            upscale=2 if upscale else None,
            aspect_ratio=aspect_ratio,
            clip_skip=clip_skip,
            is_super_user=True,
            seed=seed,
            scale=scale,
            person=(not no_person),
            no_age=no_age,
            no_race=no_race,
            pag_scale=pag_scale,
        )
        for cls, params in txt2img_iterations(
            inputParams=inputParams,
            models=models,
            category=category,
            template=template,
            template_category=template_category,
            auto_prompt=None,
            all_templates=all_templates,
            default_prompt=faceid.caption,
            default_model=image_config.face2img.default_model,
            sex=faceid.sex,
            age=faceid.age,
            race=faceid.race,
        ):
            params: PipelineParams = cls.pipelineParams(**params.model_dump())
            res = cls.from_face(
                params=params,
                faceid_embeds=faceid.embeds,
                face_path=faceid.path_crop,
            )
            cls.release()
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

    for face in faces:
        render(face)

    if len(results) > 3:
        concat_results, _ = Concat(outdir).concat_from_paths(results)
        print_term_image(image_path=concat_results, height=40)


@router.get("/face2img-options")
async def api_face2img_options():
    return {
        "safe": {
            "models": Diffusers.options_for_category("face"),
            "templates": [t.name for t in image_config.get_template_category("face")],
        },
        "nsfw": {
            "models": Diffusers.options_for_category("face-nsfw"),
            "templates": [
                t.name for t in image_config.get_template_category("face-nsfw")
            ],
        },
    }


@router.post("/face2img/{prompt}")
async def api_face2img(
    file: Annotated[UploadFile, File()],
    prompt: Annotated[str, Path(title="generator id")],
    data: Annotated[str, Form()],
):
    try:
        tmp_path = await uploaded_file(file)
        form_data = json.loads(data)
        logging.info(tmp_path)
        logging.info(form_data)
        try:
            assert form_data
        except AssertionError:
            form_data = {}
        input_params = form_data
        input_params["person"] = True
        model = form_data.get("model")
        no_person = input_params.get("no-person", None)
        if no_person:
            del input_params["no-person"]
            input_params["person"] = False
        faceid = FaceId.create(input=tmp_path, overwrite=False)
        template = input_params.get("template")
        try:
            tpl = image_config.get_template(template)
            assert tpl
            if not model:
                model = tpl.models[0]
            input_params = tpl.apply(
                input_params=input_params,
                sex=faceid.sex,
                age=faceid.age,
                race=faceid.race,
            )
        except AssertionError:
            pass
        try:
            assert model
        except AssertionError:
            model = image_config.face2img.default_model
        cls = Diffusers.cls_for_option(model)
        params = cls.pipelineParams(**input_params)
        if params.person:
            params.apply_person(sex=faceid.sex, age=faceid.age, race=faceid.race)
        if not params.prompt:
            params.prompt = faceid.caption
        params.upscale = 2
        image_result = cls.from_face(
            params=params, faceid_embeds=faceid.embeds, face_path=faceid.path_crop
        )
        assert image_result
        cls.release()
        logging.info(image_result.text)
        return make_multipart_response(
            image_path=image_result.image[0], message=image_result.text
        )
    except IncompatibleTemplate as e:
        logging.exception(e)
        raise HTTPException(415, e.__cause__)
    except Exception as e:
        logging.exception(e)
        raise HTTPException(500, e.__cause__)
