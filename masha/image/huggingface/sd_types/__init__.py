from masha.image.huggingface.sd_types.lcm import StableDiffusionLCM
from masha.image.huggingface.sd_types.opendalle import StableDiffusionOpenDalle
from masha.image.huggingface.sd_types.pixart import StableDiffusionPixArt
from .sd import StableDiffusionSD
from .kandinsky import StableDiffusionKANDINSKY
from .kandinsky3 import StableDiffusionKANDINSKY3
from .sdxl import StableDiffusionSDXL
from .sd3 import StableDiffusion3
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
            case SDModelType.KANDINSKY:
                return StableDiffusionKANDINSKY
            case SDModelType.KANDINSKY3:
                return StableDiffusionKANDINSKY3
            case SDModelType.PIXART:
                return StableDiffusionPixArt
            case SDModelType.LCM:
                return StableDiffusionLCM 
            case SDModelType.OPENDALLE:
                return StableDiffusionOpenDalle
            case _:
                raise AssertionError

    for sd in image_config.txt2img:
        cls = sd.model if isinstance(sd.model, str) else sd.model[0]
        if sd.option != sd.model:
            cls += f'_{sd.option}'
        inst = type(cls, (get_model_type_class(sd.sd_type),), sd.model_dump())
        inst.register(sd.model)
