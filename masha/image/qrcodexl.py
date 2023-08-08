from enum import StrEnum
from functools import reduce
import logging
from pathlib import Path
from typing import Any, Optional
from masha.image.config import ParamsConfig, image_config
from os import environ
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
)
import torch
from PIL import Image
from coreimage.qrcode import get_qrcode, ERROR
from corestring import to_token, truncate, clean_newlines
from corefile import TempPath
from masha.image.diffusers import Diffusers
from masha.image.huggingface.utils import get_compel_prompts_xl, load_image
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
from masha.image.config import image_config
import shlex
from rich import print
from masha.image.classify import Classifier

from masha.image.huggingface.lora.sdxl_loaders import LoadersSDXLMixin


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
    def templates(cls) -> list[str]:
        return [k for k in cls.classes.keys()]

    @property
    def templates_enum(cls) -> StrEnum:
        return StrEnum("models", cls.templates)

    def upscale(cls, img: Image.Image) -> Image.Image:
        return Upscale.upscale_img(img, scale=2)

    def get_template(cls, name) -> "QRCodeMeta":
        k = name.lower()
        assert k in cls.classes
        return cls.classes[k]

    def get_qrcode_image(cls, data: list[str]) -> Image.Image:
        code_image = get_qrcode(data, box_area=16, border=4, error=ERROR.EXTREME.value)
        w = code_image.width + 128
        image = Image.new("RGB", (w, w), (128, 128, 128))
        position = (
            (image.width - code_image.width) // 2,
            (image.height - code_image.height) // 2,
        )
        image.paste(code_image, position)
        logging.debug(f">> generating qrcode for {data}")
        try:
            print("")
            print_term_image(image=image, height=20)

        except Exception:
            pass
        return image

    def split_qr(cls, qr: str) -> list[str]:
        def clean_split(res, x):
            if any([not len(res), "=" in x]):
                res.append(x)
            else:
                res[-1] += f" {x}"
            return res

        return reduce(clean_split, shlex.split(qr), [])

    def txt2qr(cls, data: list[str] | str) -> Path:
        if isinstance(data, str):
            data = cls.split_qr(data)
        qr = cls.get_qrcode_image(data)
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


class QRCode(LoadersSDXLMixin, metaclass=QRCodeMeta):
    name: str
    control_net: str
    model: str
    controlnet_conditioning_scale = None
    guidance_scale = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = None
    seed: Optional[int] = None
    control_guidance_start = None
    control_guidance_end = None
    strength = None
    scheduler_class: Optional[str] = None
    params = None

    def __del__(self):
        self.do_release()

    def do_release(self):
        logging.info("Releasing cache")
        self.pipe = None
        self.image_pipe = None
        gc.collect()
        empty_cache()
        logging.info(f"Memory allocated - {format_size(current_allocated_memory())}")

    @property
    def text_inversion_root(self) -> Path:
        return self.__class__.dataRoot / image_config.textual_inversion.root_xl

    @property
    def lora_path(self) -> Path:
        return Diffusers.loraxlPath

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

    def get_control_net(self) -> ControlNetModel:
        path = self.controlPath
        if path.is_file():
            return ControlNetModel.from_single_file(
                path.as_posix(),
                torch_dtype=torch.float16,
            )
        return ControlNetModel.from_pretrained(
            path.as_posix(),
            torch_dtype=torch.float16,
        )

    def get_prompt(self, prompt: str, image: Optional[Path] = None) -> str:
        try:
            result = ",".join([prompt,self.prompt]).strip()
            assert not result
            assert image
            assert image.exists()
            clssifiers = Classifier.classify(image=image)
            assert len(clssifiers)
            logging.debug(f"classifiers -> {clssifiers}")
            return " ".join([c.label for c in clssifiers])
        except AssertionError:
            return result

    def do_code2img(self, code: Image.Image, **kwds) -> Path:
        args = {k: v for k, v in kwds.items() if v}
        base_size = 512
        control_image_small = center_crop_resize(code, (base_size, base_size))
        control_image_large = center_crop_resize(code, (base_size * 2, base_size * 2))
        image: Path = args.get("image")
        self.model = args.get("model", self.model)
        guidance_scale = args.get(
            "guidance_scale", self.guidance_scale or self.modelParams.guidance_scale
        )
        controlnet_conditioning_scale = args.get(
            "controlnet_conditioning_scale", self.controlnet_conditioning_scale
        )
        num_inference_steps = args.get(
            "num_inference_steps", self.num_inference_steps or self.modelParams.num_inference_steps
        )
        seed = args.get("seed", self.seed)
        prompt = self.get_prompt(args.get("prompt", ""), image)
        negative_prompt = args.get(
            "negative_prompt", self.negative_prompt or self.modelParams.negative_prompt
        )
        auto_prompt = args.get("auto_prompt")
        strength = args.get("strength", self.strength)
        device = self.__class__.device

        logging.info(f">> MODEL: {Diffusers.cls_for_option(self.model).__name__}")
        logging.info(f">> DEVICE: {device}")
        if auto_prompt:
            prompt = Prompt.cls_for_option(str(auto_prompt)).generate(prompt)
        elif not prompt:
            prompt = Prompt.cls_for_option("1").generate("")
        self.prompt = clean_newlines(prompt)
        if negative_prompt:
            self.negative_prompt = clean_newlines(negative_prompt)
        logging.info(f">> PROMPT: {self.prompt}")
        logging.info(f">> NEGATIVE: {self.negative_prompt}")
        controlnet = self.get_control_net()
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.modelPath.as_posix(),
            controlnet=controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        try:
            assert self.scheduler_class
            scheduler = self.modelConfig.scheduler.from_config(
                config=pipe.scheduler.config
            )
            pipe.scheduler = scheduler
            logging.info(f">> SCHDULER {scheduler.__class__.__name__}")
            logging.debug(scheduler.config)
        except AssertionError:
            pass
        logging.debug(pipe.scheduler)
        self.pipe: StableDiffusionXLControlNetPipeline = pipe.to(
            device, dtype=torch.float16
        )
        try:
            self.loadLoraWeights(pipe=self.pipe, prompt=self.prompt)
            self.loadTextualInversion(
                pipe=self.pipe, prompt=self.prompt, negative_prompt=self.negative_prompt
            )
        except Exception:
            pass
        image_pipe = StableDiffusionXLControlNetImg2ImgPipeline(
            text_encoder=self.pipe.text_encoder,
            text_encoder_2=self.pipe.text_encoder_2,
            tokenizer=self.pipe.tokenizer,
            tokenizer_2=self.pipe.tokenizer_2,
            unet=self.pipe.unet,
            controlnet=self.pipe.controlnet,
            scheduler=self.pipe.scheduler,
            vae=self.pipe.vae,
        )
        try:
            self.loadLoraWeights(pipe=self.image_pipe, prompt=self.prompt)
            self.loadTextualInversion(
                pipe=self.image_pipe,
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
            )
        except Exception:
            pass
        self.image_pipe = image_pipe.to(device, dtype=torch.float16)
        if not seed:
            seed = torch.Generator(device).seed()
        generator = torch.Generator(device).manual_seed(seed)

        temp_name = truncate(to_token(prompt), size=30, ellipsis="")
        temp_path = TempPath(f"{self.name}{temp_name}{seed}{uuid4().hex}.jpg")
        logging.info(f">> SEED: {seed}")
        control_args = dict(
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
            strength=strength,
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        )
        if image and image.exists():
            image_crop = center_crop_resize(
                load_image(image), (base_size * 2, base_size * 2)
            )
            print_term_image(image=image_crop)
            result = self.image_pipe(
                control_image=control_image_large,
                image=image_crop,
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                **control_args,
            ).images[0]
        else:
            latents = self.pipe(
                image=control_image_small,
                output_type="latent",
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            )
            upscaled_latents = upscale(latents, "nearest-exact", 2)
            result = self.image_pipe(
                control_image=control_image_large,
                image=upscaled_latents,
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            ).images[0]
        upscaled = self.__class__.upscale(result)
        upscaled.save(temp_path.as_posix())
        print(control_args)
        return temp_path


class BaseQRCode(QRCode):
    name: str
    control_net: str
    model: str
    controlnet_conditioning_scale = None
    guidance_scale = None
    strength = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = None
    seed: Optional[int] = None


def register_qrcodes():
    for sd in image_config.qrcode:
        cls = sd.name
        inst = type(cls, (BaseQRCode,), sd.model_dump())
        inst.register(sd.name)
