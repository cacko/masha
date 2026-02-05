from .sd import StableDiffusionSD
from .sdxl import StableDiffusionSDXL
from .sd3 import StableDiffusion3
from .mflux import StableDiffusionMFlux
from .flux import StableDiffusionFlux
from masha.image.config import image_config
from masha.image.models import SDModelType


def register_models():

    def get_model_type_class(model_type: SDModelType):
        match model_type:
            case SDModelType.SD:
                return StableDiffusionSD
            case SDModelType.SDXL:
                return StableDiffusionSDXL
            case SDModelType.SD3:
                return StableDiffusion3
            case SDModelType.FLUX:
                return StableDiffusionFlux
            case SDModelType.MFLUX:
                return StableDiffusionMFlux
            case _:
                raise AssertionError

    for sd in image_config.txt2img:
        cls = sd.model if isinstance(sd.model, str) else sd.model[0]
        if sd.option != sd.model:
            cls += f"_{sd.option}"
        inst = type(cls, (get_model_type_class(sd.sd_type),), sd.model_dump())
        inst.register(sd.model)
