from diffusers import (
    StableDiffusionImageVariationPipeline,
)
from torchvision import transforms
from PIL import Image
from pathlib import Path
from os import environ
from typing import Optional, Any
from masha.core import perftime
from masha.core.bytes import round8
from .upscale import Upscale
from .models import VariationPipelineParams
from coreimage.organise import Concat


class VariationMeta(type):

    __data_root: Optional[Path] = None
    __model_path: Optional[Path] = None

    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        return type.__call__(cls, *args, **kwds)

    def register(cls, model: str):
        cls.__model_path = cls.dataRoot / model

    @property
    def dataRoot(self) -> Path:
        if not self.__data_root:
            self.__data_root = Path(environ.get("HUGGINGROOT", "."))
        return self.__data_root

    @property
    def modelPath(self) -> Path:
        assert self.__model_path
        return self.__model_path

    @property
    def device(cls):
        return "mps"

    @property
    def pipeline(cls) -> StableDiffusionImageVariationPipeline:
        sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            cls.modelPath.as_posix(),
            safety_checker=None,
        ).to(cls.device)
        return sd_pipe

    def variation(
        cls,
        src_path: Path,
        out_path: Path,
        params: VariationPipelineParams
    ) -> Optional[Path]:
        res, _ = cls().do_variation(src=src_path, dst=out_path, params=params)
        return res


class Variation(object, metaclass=VariationMeta):

    __pipeline: Optional[StableDiffusionImageVariationPipeline] = None

    @property
    def pipe(self) -> StableDiffusionImageVariationPipeline:
        if not self.__pipeline:
            self.__pipeline = Variation.pipeline
        return self.__pipeline

    def do_variation(
        self,
        src: Path,
        dst: Path,
        params: VariationPipelineParams
    ) -> Path:
        use_height = 512
        im = Image.open(src.as_posix())
        tform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.Normalize(
                    [0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711]
                ),
            ]
        )
        with perftime(f"Variations: {Variation.modelPath.name}"):
            inp = tform(im).to(Variation.device).unsqueeze(0)  # type: ignore
            out_images = []
            ratio = min(use_height / im.width, use_height / im.height)
            for _ in range(max(1, params.num_images_per_prompt)):
                out = self.pipe(
                    inp,
                    guidance_scale=params.guidance_scale,
                    height=round8(min(use_height, im.height * ratio)),
                    width=round8(min(use_height, im.width * ratio)),
                    num_inference_steps=params.num_inference_steps
                )
                upscaled_img = Upscale.upscale_img(out["images"][0])  # type: ignore
                upscaled_img = upscaled_img.resize(size=(im.width, im.height))
                out_images.append(upscaled_img)  # type:ignore
            return Concat(dst).concat_from_images(out_images)
