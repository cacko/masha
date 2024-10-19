from pathlib import Path
import re
from typing import Generator, Optional
from corestring import to_float
from corefile import filepath
from PIL import Image
from diffusers.utils import load_image as diffusers_load_image
import logging
from compel import Compel, ReturnedEmbeddingsType, DiffusersTextualInversionManager
from masha.image.config import EmbeddingConfig
from masha.image.models import Embedding, IncompatibleTemplate, PipelineParams, Sex
from masha.image.diffusers import Diffusers, DiffusersType
from masha.image.config import image_config
from random import shuffle
from masha.image.prompt import Prompt
from itertools import product
from masha.name.classifiers import Ethnicity
from masha.image.huggingface.lora import LORA_PATTERN

TI_PATTERN = re.compile(r"\<inversion:([^>]+):([\d.]+)\>", re.DOTALL)
EM_PATTERN = re.compile(r"embedding:([^\s,:]+)")
SCORE_SPLIT_PATTERN = re.compile(r"[\s>]+")


def get_models_score(
    prompt: str,
    root: Path,
    pattern: re.Pattern,
    ext: list[str] = [".pt", ".safetensors"],
) -> Generator[tuple[Path, float], None, None]:
    files = {fp.stem.lower(): fp for fp in filepath(root, suffixes=ext)}
    for lora, score in pattern.findall(prompt):
        logging.debug(f"LORA FOUND: {lora} {score}")
        k = lora.lower()
        if k not in files:
            continue
        fp = files.get(k)
        score = to_float(score)
        yield (fp, score)


def get_embeddings(
    prompt: str,
    root: Path,
    embeddings: list[EmbeddingConfig],
    ext: list[str] = [".pt", ".safetensors"],
) -> Generator[Embedding, None, None]:
    files = {fp.stem: fp for fp in filepath(root, suffixes=ext)}
    for config in embeddings:
        if config.name not in files:
            continue
        used_embeddings = list(
            filter(lambda e: re.search(rf"\b{e}\b", prompt, re.I | re.M), config.token)
        )
        if not len(used_embeddings):
            continue
        logging.debug(f"Embeddings FOUND: {config.name} -> {used_embeddings}")
        fp = files.get(config.name)
        yield from [Embedding(path=fp, token=e) for e in used_embeddings]


def get_lora_models(
    prompt: str,
    root: Path,
    ext: list[str] = [".pt", ".safetensors"],
) -> Generator[tuple[Path, float], None, None]:
    yield from get_models_score(prompt=prompt, root=root, pattern=LORA_PATTERN, ext=ext)


def get_ti_models(
    prompt: str,
    root: Path,
    ext: list[str] = [".pt", ".safetensors"],
) -> Generator[tuple[Path, float], None, None]:
    logging.warn(f"{root} {prompt}")
    yield from get_models_score(prompt=prompt, root=root, pattern=TI_PATTERN, ext=ext)


def load_image(image_path: Path, size: Optional[tuple[float]] = None) -> Image.Image:
    image = diffusers_load_image(image_path.as_posix())
    if size:
        image.thumbnail(size)
    return image


def get_compel_prompts(pipe, prompt=None, negative_prompt=None):
    prompt_embeds, negative_prompt_embeds = None, None
    textual_inversion_manager = DiffusersTextualInversionManager(pipe)
    compel = Compel(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        truncate_long_prompts=True,
        textual_inversion_manager=textual_inversion_manager,
    )
    if prompt:
        parsed = compel.parse_prompt_string(prompt)
        prompt_embeds, _ = compel.build_conditioning_tensor_for_conjunction(parsed)
    if negative_prompt:
        negative_prompt_embeds = compel(negative_prompt)
    if all([prompt, negative_prompt]):
        [
            prompt_embeds,
            negative_prompt_embeds,
        ] = compel.pad_conditioning_tensors_to_same_length(
            [prompt_embeds, negative_prompt_embeds]
        )
    return prompt_embeds, negative_prompt_embeds


def get_compel_prompts_xl(pipe, prompt=None, negative_prompt=None):
    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
    )
    (
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = (
        None,
        None,
        None,
        None,
    )
    if prompt:
        prompt_embeds, pooled_prompt_embeds = compel(prompt)
    if negative_prompt:
        negative_prompt_embeds, negative_pooled_prompt_embeds = compel(negative_prompt)
    if all([prompt, negative_prompt]):
        [
            prompt_embeds,
            negative_prompt_embeds,
        ] = compel.pad_conditioning_tensors_to_same_length(
            [prompt_embeds, negative_prompt_embeds]
        )
    return (
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    )


def txt2img_iterations(
    inputParams: dict,
    models: Optional[str],
    category: Optional[str],
    template: Optional[str],
    template_category: Optional[str],
    auto_prompt: Optional[str],
    all_prompts: bool = False,
    all_templates: bool = False,
    default_prompt: str = None,
    default_model: str = None,
    sex: Optional[Sex] = None,
    age: Optional[int] = None,
    race: Optional[Ethnicity] = None,
    throw_exception = False,
) -> Generator[tuple[DiffusersType, PipelineParams], None, None]:
    auto_prompts: list[Optional[str]] = [auto_prompt]
    auto_templates: list[Optional[str]] = [None]
    prompt = inputParams.get("prompt")
    try:
        assert template
        tpl = image_config.get_template(template)
        if not models:
            models = tpl.models
        assert tpl
        auto_templates = [template]
    except AssertionError:
        pass
    try:
        assert template_category
        templates = image_config.get_template_category(template_category)
        assert templates
        auto_templates += [t.name for t in templates]
        auto_templates.remove(None)
    except AssertionError:
        pass
    if category:
        models = (
            Diffusers.options[:]
            if category == "all"
            else Diffusers.options_for_category(category)
        )
        shuffle(models)
        logging.info(f"Using models: {models}")
    if all_prompts:
        auto_prompts = list(Prompt.options)
    if all_templates:
        auto_templates = Diffusers.templates
    if not models:
        models = [None]
    for mdl, current_auto_prompt, current_template in product(
        models, auto_prompts, auto_templates
    ):
        try:
            assert current_template
            tpl = image_config.get_template(current_template)
            if not mdl:
                mdl = tpl.models[0]
            logging.info(f"LOAD TEMPLATE -> {current_template}")
            cls = Diffusers.cls_for_option(mdl, scheduler_class=tpl.scheduler_class)
            assert cls
            params = cls.pipelineParams(
                **tpl.apply(input_params=inputParams, sex=sex, age=age, race=race),
                model=mdl,
                auto_prompt=current_auto_prompt,
            )
            print(params)
            if not params.prompt:
                params.prompt = default_prompt
            logging.debug(params)
            yield (cls, params)
        except AssertionError:
            if not mdl:
                mdl = default_model
            if not prompt:
                inputParams["prompt"] = default_prompt
            cls = Diffusers.cls_for_option(mdl)
            params = cls.pipelineParams(
                **inputParams,
                model=mdl,
                auto_prompt=current_auto_prompt,
            )

            params.apply_person(sex=sex, age=age, race=race)
            yield (cls, params)
        except IncompatibleTemplate as e:
            if throw_exception:
                raise e
            logging.error("Incompatible template")
            continue


def img2img_iterations(
    inputParams: dict,
    models: Optional[list[str]] = [None],
    category: Optional[str] = None,
    style: Optional[str] = None,
    all_styles: bool = False,
    template: Optional[str] = None,
    all_templates: bool = False,
    template_category: str = None,
) -> Generator[tuple[DiffusersType, PipelineParams], None, None]:
    params = None
    auto_styles: list[Optional[str]] = [None]
    auto_templates: list[Optional[str]] = [None]
    try:
        assert style
        stl = image_config.get_style(style)
        if all([x is None for x in models]):
            models = [stl.model]
        inputParams = stl.apply(input_params=inputParams)
    except AssertionError:
        pass
    try:
        assert template
        tpl = image_config.get_template(template)
        if not models:
            models = tpl.models
        assert tpl
        auto_templates = [template]
    except AssertionError:
        pass
    try:
        assert template_category
        templates = image_config.get_template_category(template_category)
        assert templates
        auto_templates += [t.name for t in templates]
        auto_templates.remove(None)
    except AssertionError:
        pass
    logging.debug(models)
    if category:
        models = (
            Diffusers.options[:]
            if category == "all"
            else Diffusers.options_for_category(category)
        )
        shuffle(models)
        logging.info(f"Using models: {models}")
    if all_styles:
        auto_styles = image_config.style_options
    if all_templates:
        auto_templates = Diffusers.templates
    for mdl, current_style, current_template in product(
        models, auto_styles, auto_templates
    ):
        params = None
        try:
            assert current_style
            stl = image_config.get_style(current_style)
            assert stl
            if not mdl:
                mdl = stl.model
            cls = Diffusers.cls_for_option(mdl)
            params = cls.pipelineParams(
                **{
                    **stl.apply(input_params=inputParams),
                    **dict(model=mdl),
                },
            )
        except AssertionError:
            pass

        try:
            assert current_template
            print()
            tpl = image_config.get_template(current_template)
            assert tpl
            if not mdl:
                mdl = tpl.models[0]
            cls = Diffusers.cls_for_option(mdl)
            params = cls.pipelineParams(
                **tpl.apply(input_params=inputParams), model=mdl
            )
        except AssertionError:
            pass

        try:
            assert not params
            cls = Diffusers.cls_for_option(mdl)
            params = cls.pipelineParams(
                **{
                    **inputParams,
                    **dict(model=mdl),
                }
            )

        except AssertionError:
            pass
        yield (cls, params)
