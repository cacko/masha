import logging
from PIL import Image
from pathlib import Path
from os import environ
from typing import Optional, Any
from pydantic import BaseModel
import torch
from PIL.ExifTags import Base as TagNames
from corefile import TempPath
import numpy as np
from masha.core import perftime
from spandrel import ImageModelDescriptor, ModelLoader
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
import cv2

# def to_pil_image(tensor: torch.Tensor) -> Image.Image:
#     image = tensor.cpu().squeeze().detach().numpy()
#     image = np.transpose(image, (1, 2, 0))
#     image = np.clip((image * 255.0).round(), 0, 255)
#     image = image.astype(np.uint8)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     return Image.fromarray(image)


class UpscaleMeta(type):
    __data_root: Optional[Path] = None
    __models_root: Optional[Path] = None
    __models = {
        2: "RealESRGAN_x2plus_mtg_v1.pth",
        4: "RealESRGAN_x4plus.pth",
        8: "RealESRGAN_x8.pth",
        20: "BSRGANx2.pth",
        40: "BSRGAN.pth",
        400: "Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth",
        1000: "scunet_color_real_psnr.pth",
        1001: "Uformer_B.pth",
    }

    def get_upscaler(cls, scale: int) -> ImageModelDescriptor:
        model_path = cls.__models_root / cls.__models[scale]
        model = ModelLoader().load_from_file(model_path.as_posix())
        model.to(cls.device)
        model.eval()
        return model

    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        return type.__call__(cls, *args, **kwds)

    def register(cls, root: str):
        cls.__models_root = cls.dataRoot / root

    @property
    def dataRoot(cls) -> Path:
        if not cls.__data_root:
            cls.__data_root = Path(environ.get("HUGGINGROOT", "."))
        return cls.__data_root

    @property
    def device(cls):
        return torch.device("mps")

    def upscale(
        cls, src_path: Path, dst_path: Optional[Path] = None, **kwds
    ) -> Optional[Path]:
        if not dst_path:
            dst_path = TempPath(f"{src_path.stem}.png")
        with perftime(f"{cls.__name__}"):
            res = cls().do_upscale(src=src_path, dst=dst_path, **kwds)
            return dst_path if res else None

    def upscale_img(cls, img: Image.Image, **kwds) -> Image.Image:
        return cls().do_upscale_img(img, **kwds)


class Upscale(object, metaclass=UpscaleMeta):
    def do_upscale_img(self, low_res_img, **kwds):
        with torch.no_grad():
            scale = kwds.get("scale", 4)
            small_tensor = to_tensor(low_res_img).unsqueeze(0).to(self.__class__.device)
            upscaler = self.__class__.get_upscaler(scale)
            upscaled_tensor = upscaler(small_tensor)
            return to_pil_image(upscaled_tensor.squeeze())

    @staticmethod
    def set_info(image: Image.Image, prompt) -> Image.Exif:
        ex = image.getexif()
        ex[TagNames.ImageDescription] = prompt
        return ex

    def do_upscale(self, src: Path, dst: Path, **kwds) -> bool:
        scale = kwds.get("scale", 4)
        low_res_img = Image.open(src.as_posix()).convert("RGB")

        upscaled = self.do_upscale_img(low_res_img, scale=scale)

        prompt = None
        try:
            ex = low_res_img.getexif()
            prompt = ex.get(TagNames.ImageDescription)
        except Exception as e:
            logging.exception(e)

        if prompt:
            upscaled.save(
                dst.as_posix(),
                optimize=False,
                exif=Upscale.set_info(image=upscaled, prompt=prompt),
            )
        else:
            upscaled.save(dst.as_posix())
        return True
