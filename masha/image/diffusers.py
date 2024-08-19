from enum import StrEnum
import json
from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
    DDIMScheduler,
    DPMSolverSDEScheduler,
    DEISMultistepScheduler,
    DPMSolverSinglestepScheduler,
    ScoreSdeVeScheduler,
    SchedulerMixin,
    LCMScheduler,
    KDPM2AncestralDiscreteScheduler,
    EDMDPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
)
import rich
from masha.config import app_config
from queue import Queue
from pathlib import Path
from os import environ
from typing import Optional, Any
from pydantic import BaseModel
from masha.image.config import image_config
from PIL import Image
from masha.image.upscale import Upscale
from masha.image.models import ImageResult, PipelineParams, PipelineType, Resolutions
import diffusers
from faker import Faker
from faker.factory import logger
from stringcase import snakecase
import logging
from itertools import chain

diffusers.logging.set_verbosity_error()
logger.setLevel(logging.INFO)

AVAILABLE_MEMORY_THRESHOLD = 4843545600


class IMAGE_FORMAT(StrEnum):
    JPEG = "jpg"
    PNG = "png"
    WEBP = "webp"


class PipeItem(BaseModel, arbitrary_types_allowed=True):
    pipe: DiffusionPipeline
    model: Path


class Config(BaseModel):
    model: str
    option: str
    category: list[str]
    img2img_model: Optional[str] = None


class DiffusersType(type):
    is_superuser: bool = False
    image_format: IMAGE_FORMAT = IMAGE_FORMAT.WEBP
    configs: dict[str, Config] = {}
    classes: dict[str, "DiffusersType"] = {}
    pipe_item: Optional[PipeItem] = None
    pipe_interupt = False
    __queue: Optional[Queue] = None
    __faker: Optional[Faker] = None
    __instances: dict[str, "DiffusersType"] = {}

    def __call__(cls, **kwds: Any) -> Any:
        k = cls.__name__
        if k not in cls.__instances:
            cls.__instances[k] = type.__call__(cls, **kwds)
        return cls.__instances[k]

    def register(cls, model: str):
        cfg = image_config.get_sd_config(model)
        cls.configs[cls.__name__] = Config(
            model=model,
            option=cfg.option,
            category=[c.strip() for c in cfg.category.split(",")],
        )
        logging.debug(f">>> Registered model {model} as {cls}")
        cls.classes[cls.__name__] = cls

    @property
    def queue(cls) -> Queue:
        if not cls.__queue:
            cls.__queue = Queue()
        return cls.__queue

    @property
    def faker_instance(cls) -> Faker:
        if not cls.__faker:
            cls.__faker = Faker()
        return cls.__faker

    def get_filestem(cls, params: PipelineParams) -> str:
        rand = snakecase(cls.faker_instance.text(max_nb_chars=30).strip("."))
        res = f"{params.output_width}x{params.output_height}"
        logging.warning(f"GET_FILESYSTEM -> {rand}_{res}")
        
        return f"{rand}_{res}"

    @property
    def dataRoot(cls) -> Path:
        return Path(environ.get("HUGGINGROOT", ".")) / "txt2img"

    @property
    def modelPath(cls) -> Path:
        return cls.dataRoot / cls.configs[cls.__name__].model

    @property
    def img2imgModelPath(cls) -> Optional[Path]:
        cfg = cls.configs[cls.__name__]
        try:
            assert cfg.img2img_model
            return cls.dataRoot / cfg.img2img_model
        except:
            return cls.modelPath

    @property
    def lcmPath(cls) -> Optional[Path]:
        try:
            assert cls.lcm
            return cls.dataRoot / cls.lcm
        except AssertionError:
            return None

    @property
    def pipelineClass(cls) -> PipelineType:
        try:
            cfg_path = cls.modelPath / "model_index.json"
            assert cfg_path.exists()
            fg = json.loads(cfg_path.read_text())
            return PipelineType(fg.get("_class_name"))
        except AssertionError:
            return PipelineType.COREML

    @property
    def is_v2(cls) -> bool:
        name = cls.modelPath.name
        yaml = cls.modelPath / f"{name}.yaml"
        return yaml.exists()

    @property
    def controlPath(cls) -> Path:
        return cls.dataRoot / app_config.image.controlnet_model

    @property
    def loraPath(cls) -> Path:
        return cls.dataRoot / image_config.lora.root

    @property
    def loraxlPath(cls) -> Path:
        return cls.dataRoot / image_config.lora.root_xl

    @property
    def textualInversionRoot(cls) -> Path:
        match cls.pipelineClass:
            case PipelineType.SDXL:
                return
            case PipelineType.SD:
                if cls.is_v2:
                    return cls.dataRoot / image_config.textual_inversion.root_2
                return cls.dataRoot / image_config.textual_inversion.root
        return cls.dataRoot / image_config.textual_inversion.root_2

    @property
    def device(cls):
        return environ.get("DEVICE", "mps")

    def pipelineParams(cls, **params) -> PipelineParams:
        return cls().get_pipelineParams(params)

    @property
    def option(cls) -> str:
        return cls.configs.get(cls.__name__).option

    @property
    def options(cls) -> list[str]:
        return [x.option for x in cls.configs.values()]

    @property
    def options_enum(cls) -> StrEnum:
        return StrEnum("options", cls.options)

    @property
    def templates(cls) -> list[str]:
        return [x.name for x in image_config.templates]

    @property
    def templates_enum(cls) -> StrEnum:
        return StrEnum("templates", cls.templates)

    @property
    def resolutions_enum(cls) -> StrEnum:
        return StrEnum("resolutions", Resolutions.keys())

    @property
    def categories(cls) -> list[str]:
        return list(
            set(
                list(chain.from_iterable([[*x.category] for x in cls.configs.values()]))
            )
        )

    @property
    def categories_enum(cls) -> StrEnum:
        return StrEnum("categories", cls.categories)

    def options_for_category(cls, category: str) -> list[str]:
        return list(
            map(
                lambda c: c.option,
                filter(lambda cfg: category in cfg.category, cls.configs.values()),
            )
        )

    def cls_for_option(
        cls,
        opt: str,
        scheduler_class: Optional[str] = None,
        scheduler_args: Optional[dict[str, Any]] = None,
    ) -> "DiffusersType":
        res = next(
            filter(lambda x: cls.configs[x].option == opt, cls.configs.keys()), None
        )
        assert res
        instance = cls.classes[res]
        if scheduler_class:
            instance.scheduler_class = scheduler_class
        if scheduler_args:
            instance.sceheduler_args = scheduler_args
        return instance

    def upscale(cls, img: Image.Image, scale: Optional[int] = None) -> Image.Image:
        return Upscale.upscale_img(img, scale=scale)

    @property
    def image_suffix(cls) -> str:
        return f"{cls.image_format.value}"

    def from_text(cls, params: PipelineParams) -> Optional[ImageResult]:
        try:
            cls.pipe_interupt = False
            return cls().generate_from_text(params)
        except KeyboardInterrupt:
            cls.pipe_interupt = True

    def from_img(
        cls, img_path: Path, params: PipelineParams, **kwds
    ) -> Optional[ImageResult]:
        try:
            cls.pipe_interupt = False
            return cls().generate_from_image(img_path, params, **kwds)
        except KeyboardInterrupt:
            cls.pipe_interupt = True

    def from_face(
        cls, faceid_embeds, params: PipelineParams, **kwds
    ) -> Optional[ImageResult]:
        try:
            cls.pipe_interupt = False
            return cls().generate_from_face(faceid_embeds, params, **kwds)
        except KeyboardInterrupt:
            cls.pipe_interupt = True

    def release(cls):
        cls.__instances[cls.__name__].do_release()
        cls.__instances[cls.__name__] = None
        del cls.__instances[cls.__name__]

    def interrupt_callback(cls, pipe, i, t, callback_kwargs):
        if cls.pipe_interupt:
            pipe._interrupt = True
        return callback_kwargs


class Diffusers(object, metaclass=DiffusersType):
    option = "default"
    category = "general"
    scheduler_class = "EulerAncestralDiscreteScheduler"
    scheduler_args: dict[str, any] = {}

    @property
    def scheduler(self):
        try:
            scheduler = self.scheduler_class
            if isinstance(scheduler, SchedulerMixin):
                return scheduler
            match scheduler:
                case "EulerAncestralDiscreteScheduler":
                    return EulerAncestralDiscreteScheduler
                case "EulerDiscreteScheduler":
                    return EulerDiscreteScheduler
                case "DPMSolverMultistepScheduler":
                    return DPMSolverMultistepScheduler
                case "DPMSolverSinglestepScheduler":
                    return DPMSolverSinglestepScheduler
                case "DEISMultistepScheduler":
                    return DEISMultistepScheduler
                case "DDIMScheduler":
                    return DDIMScheduler
                case "UniPCMultistepScheduler":
                    return UniPCMultistepScheduler
                case "DPMSolverSDEScheduler":
                    return DPMSolverSDEScheduler
                case "ScoreSdeVeScheduler":
                    return ScoreSdeVeScheduler
                case "LCMScheduler":
                    return LCMScheduler
                case "KDPM2AncestralDiscreteScheduler":
                    return KDPM2AncestralDiscreteScheduler
                case "EDMDPMSolverMultistepScheduler":
                    return EDMDPMSolverMultistepScheduler
                case "LMSDiscreteScheduler":
                    return LMSDiscreteScheduler
                case _:
                    return EulerAncestralDiscreteScheduler
        except AssertionError:
            return None
