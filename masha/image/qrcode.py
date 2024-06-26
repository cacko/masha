from curses import A_ALTCHARSET
from enum import StrEnum
from functools import reduce
import logging
from pathlib import Path
from typing import Any, Optional
import rich
from masha.image.config import ParamsConfig, image_config
from os import environ
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    LCMScheduler,
)
import torch
from PIL import Image
from coreimage.qrcode import get_qrcode, ERROR
from corestring import clean_newlines, to_token, truncate
from corefile import TempPath
from masha.image.diffusers import Diffusers
from masha.image.huggingface.utils import get_compel_prompts
from masha.image.prompt import Prompt
import gc
from torch.mps import (
    empty_cache,
    current_allocated_memory,
)
from humanfriendly import format_size
from masha.image.upscale import Upscale
from uuid import uuid4
from coreimage.terminal import print_term_image
from masha.image.huggingface.utils import get_ti_models
from masha.image.config import image_config
import shlex
from masha.image.huggingface.lora.sd_loaders import (
    loadTextualInversion,
    loadLoraWeights,
)


def common_upscale(samples, width, height, upscale_method, crop=False):
    if crop == "center":
        old_width = samples.shape[3]
        old_height = samples.shape[2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples[:, :, y : old_height - y, x : old_width - x]
    else:
        s = samples

    return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)


def upscale(samples, upscale_method, scale_by):
    # s = samples.copy()
    width = round(samples["images"].shape[3] * scale_by)
    height = round(samples["images"].shape[2] * scale_by)
    s = common_upscale(samples["images"], width, height, upscale_method, "disabled")
    return s


def center_crop_resize(img, output_size=(512, 512)):
    width, height = img.size

    # Calculate dimensions to crop to the center
    new_dimension = min(width, height)
    left = (width - new_dimension) / 2
    top = (height - new_dimension) / 2
    right = (width + new_dimension) / 2
    bottom = (height + new_dimension) / 2

    # Crop and resize
    img = img.crop((left, top, right, bottom))
    img = img.resize(output_size)
    return img


class QRCodeMeta(type):

    classes: dict[str, "QRCodeMeta"] = {}

    def __call__(cls, **kwds: Any) -> Any:
        return type.__call__(cls, **kwds)

    def register(cls, name: str):
        logging.debug(f">>> Registered model {name} as {cls}")
        cls.classes[name] = cls

    @property
    def dataRoot(cls) -> Path:
        return Path(environ.get("HUGGINGROOT", ".")) / "txt2img"

    @property
    def models(cls) -> list[str]:
        return Diffusers.options_for_category("qr")

    @property
    def models_enum(cls) -> StrEnum:
        return StrEnum("models", cls.models)

    @property
    def templates(cls) -> list[str]:
        return [k for k in cls.classes.keys()]

    @property
    def templates_enum(cls) -> StrEnum:
        return StrEnum("templates", cls.templates)

    def upscale(cls, img: Image.Image) -> Image.Image:
        return Upscale.upscale_img(img, scale=2)

    def get_template(cls, name) -> "QRCodeMeta":
        k = name.lower()
        assert k in cls.classes
        return cls.classes[k]

    def split_qr(cls, qr: str) -> list[str]:
        def clean_split(res, x):
            if any([not len(res), "=" in x]):
                res.append(x)
            else:
                res[-1] += f" {x}"
            return res

        return reduce(clean_split, shlex.split(qr), [])

    @property
    def textualInversionRoot(cls) -> Path:
        return cls.dataRoot / image_config.textual_inversion.root

    @property
    def loraRoot(cls) -> Path:
        return cls.dataRoot / image_config.lora.root

    def get_qrcode_image(cls, data: list[str], no_padding=False, **kwds) -> Image.Image:
        code_image = get_qrcode(
            data,
            box_area=kwds.get("box_area", 16),
            border=kwds.get("border", 4),
            error=ERROR.EXTREME.value,
        )
        w = code_image.width + 128
        if no_padding:
            w = code_image.width
        image = Image.new("RGB", (w, w), (128, 128, 128))
        position = (
            (image.width - code_image.width) // 2,
            (image.height - code_image.height) // 2,
        )
        if no_padding:
            position = (0, 0)
        image.paste(code_image, position)
        logging.debug(f">> generating qrcode for {data}")
        try:
            print_term_image(image=image, height=20)

        except Exception:
            pass
        return image

    def txt2qr(cls, data: list[str] | str) -> Path:
        if isinstance(data, str):
            data = cls.split_qr(data)
        qr = cls.get_qrcode_image(data, no_padding=True, border=2)
        name = to_token("_".join(data))
        result = TempPath(f"{name}.jpg")
        upscaled = Upscale.upscale_img(qr, scale=4)
        upscaled.save(result.as_posix())
        return result

    def code2img(cls, qr: str | Image.Image, **kwds) -> Path:
        if isinstance(qr, str):
            qr = cls.get_qrcode_image(cls.split_qr(qr))
        return cls().do_code2img(code=qr, **kwds)

    @property
    def device(cls):
        return environ.get("DEVICE", "mps")

    def release(cls):
        cls().do_release()


class QRCode(object, metaclass=QRCodeMeta):
    name: str = ""
    control_net: str = ""
    model: str = ""
    controlnet_conditioning_scale = None
    guidance_scale = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = None
    seed: Optional[int] = None
    control_guidance_start = None
    control_guidance_end = None
    scheduler_class: Optional[str] = None
    clip_skip: Optional[int] = None

    def __del__(self):
        self.do_release()

    def do_release(self):
        logging.debug("Releasing cache")
        try:
            self.pipe = None
            self.image_pipe = None
            del self.pipe
            del self.image_pipe
        except AttributeError:
            pass
        gc.collect()
        empty_cache()
        logging.debug(f"Memory allocated - {format_size(current_allocated_memory())}")

    @property
    def modelPath(self) -> Path:
        return self.modelConfig.modelPath

    @property
    def modelParams(self) -> ParamsConfig:
        return ParamsConfig(**self.modelConfig.custom_params)

    @property
    def modelConfig(self) -> Diffusers:
        return Diffusers.cls_for_option(
            self.model, scheduler_class=self.scheduler_class
        )

    @property
    def controlPath(self) -> Path:
        return self.__class__.dataRoot / self.control_net

    def do_code2img(self, code: Image.Image, **kwds) -> Path:
        args = {k: v for k, v in kwds.items() if v}
        base_size = 512
        control_image_small = center_crop_resize(code, (base_size, base_size))
        control_image_large = center_crop_resize(code, (base_size * 2, base_size * 2))
        self.model = args.get("model", self.model)
        self.guidance_scale = args.get(
            "guidance_scale", self.guidance_scale or self.modelParams.guidance_scale
        )
        self.controlnet_conditioning_scale = args.get(
            "controlnet_conditioning_scale", self.controlnet_conditioning_scale
        )
        self.num_inference_steps = args.get(
            "num_inference_steps",
            self.num_inference_steps or self.modelParams.num_inference_steps,
        )
        self.clip_skip = args.get(
            "clip_skip", self.clip_skip or self.modelParams.clip_skip
        )
        seed = args.get("seed", self.seed)
        prompt = ",".join([args.get("prompt", ""), self.prompt]).strip()
        negative_prompt = args.get("negative_prompt", self.negative_prompt)
        auto_prompt = args.get("auto_prompt")
        device = self.__class__.device

        logging.info(f">> DEVICE: {device}")
        if auto_prompt:
            prompt = Prompt.cls_for_option(str(auto_prompt)).generate(prompt)
        elif not prompt:
            prompt = Prompt.cls_for_option("1").generate("")
        self.prompt = clean_newlines(prompt).strip(" ,")
        logging.info(f">> PROMPT: {self.prompt}")
        if negative_prompt:
            self.negative_prompt = clean_newlines(negative_prompt)
            logging.info(f">> NEGATIVE PROMPT: {self.negative_prompt}")
        controlnet = ControlNetModel.from_pretrained(
            self.controlPath.as_posix(),
            torch_dtype=torch.float16,
        )
        try:
            assert self.modelPath.is_file()
            pipe = StableDiffusionControlNetPipeline.from_single_file(
                self.modelPath.as_posix(),
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
                use_safetensors=True,
            )
        except AssertionError:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.modelPath.as_posix(),
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
                use_safetensors=True,
            )
        try:
            assert self.modelConfig.scheduler
            scheduler = self.modelConfig().scheduler.from_config(
                config=pipe.scheduler.config, **self.modelConfig.scheduler_args
            )
            pipe.scheduler = scheduler  # type: ignore
        except AssertionError:
            pass

        logging.info(f">> SCHDULER {pipe.scheduler.__class__.__name__}")
        self.pipe = pipe.to(device, dtype=torch.float16)

        image_pipe = StableDiffusionControlNetImg2ImgPipeline(**self.pipe.components)
        image_pipe.scheduler = self.pipe.scheduler
        self.image_pipe = image_pipe.to(device, dtype=torch.float32)
        self.image_pipe.enable_attention_slicing()
        self.image_pipe.enable_vae_slicing()
        self.image_pipe.enable_vae_tiling()
        self.pipe = loadLoraWeights(
            pipeline=self.pipe, prompt=self.prompt, lora_path=self.__class__.loraRoot
        )
        self.pipe = loadTextualInversion(
            pipeline=self.pipe,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt or "",
            text_inversion_root=self.__class__.textualInversionRoot,
        )
        prompt_embeds, negative_prompt_embeds = get_compel_prompts(
            pipe=self.pipe, prompt=self.prompt, negative_prompt=self.negative_prompt
        )
        if not seed:
            seed = torch.Generator(device).seed()
        generator = torch.Generator(device).manual_seed(seed)

        temp_name = truncate(to_token(prompt), size=30, ellipsis="")
        temp_path = TempPath(f"{self.name}{temp_name}{seed}{uuid4().hex}.jpg")
        logging.info(f">> SEED: {seed}")
        control_params = dict(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            guidance_scale=self.guidance_scale,
            controlnet_conditioning_scale=float(self.controlnet_conditioning_scale),
            clip_skip=self.clip_skip,
        )
        latents = self.pipe(
            image=control_image_small, output_type="latent", **control_params
        )
        upscaled_latents = upscale(latents, "nearest-exact", 2)
        result = self.image_pipe(
            control_image=control_image_large, image=upscaled_latents, **control_params
        ).images[0]
        rich.print(
            {
                "prompt": self.prompt,
                "negative_prompt": self.negative_prompt,
                **control_params,
            }
        )
        upscaled = self.__class__.upscale(result)
        upscaled.save(temp_path.as_posix())
        return temp_path


class BaseQRCode(QRCode):
    name: str = ""
    control_net: str = ""
    model: str = ""
    controlnet_conditioning_scale = 0.7
    guidance_scale: Optional[float] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = None
    seed: Optional[int] = None
    clip_skip: Optional[int] = None


def register_qrcodes():
    for sd in image_config.qrcode:
        cls = sd.name
        inst = type(cls, (BaseQRCode,), sd.model_dump())
        inst.register(sd.name)
