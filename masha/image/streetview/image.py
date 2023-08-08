from cachable.models import BinaryStruct
from cachable.storage import FileStorage
from cachable.storage.filestorage.image import CachableFileImage
from typing import Optional
from corestring import string_hash
from urllib.parse import urlencode, urlparse, parse_qs

from masha.image.streetview.s3 import S3
from .panorama import Panorama
from .models import Tile
from pathlib import Path
from PIL import Image
from masha.config import app_config
import logging
import concurrent.futures
from rich import print
from typing import Union
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from equilib import Equi2Cube
from corefile import TempPath
import cv2


class StreetViewImage(CachableFileImage):
    _filename: Optional[str] = None

    def __init__(self, url: str):
        self._url = url

    @property
    def store_key(self):
        return f"{self.__class__.__name__}/{self.filename[:2]}/{self.filename}"

    @property
    def storage(self):
        return FileStorage

    def tocache(self, res: BinaryStruct):
        assert self._path
        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True)
        self._path.write_bytes(res.binary)
        S3.upload(self._path, self.filename)

    @property
    def filename(self):
        parsed = urlparse(self._url)
        params = parse_qs(parsed.query)
        if "session" in params:
            del params["session"]
        query = urlencode(params, doseq=True)
        url_clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{query}"
        return string_hash(url_clean)

    @property
    def url(self):
        return self._url
    
class StreetViewTileImage(StreetViewImage):

    def tocache(self, res: BinaryStruct):
        assert self._path
        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True)
        self._path.write_bytes(res.binary)


class PanoramaImage(CachableFileImage):
    _filename: Optional[str] = None

    def __init__(self, panorama: Panorama):
        self._panorama = panorama
        print(self._panorama.metadata)

    @property
    def store_key(self):
        return f"{self.__class__.__name__}/{self.filename}"

    @property
    def storage(self):
        return FileStorage

    def tocache(self, src_path: Path):
        assert self._path
        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True)
        with Image.open(src_path) as image:
            image.save(self._path)

    @property
    def filename(self):
        return f"{self._panorama.id}.webp"

    def _init(self):
        if self.isCached:
            return
        try:
            pan_path = self.__fetch_panorama()
            self.tocache(src_path=pan_path)
        except Exception as e:
            logging.exception(e)

    def __fetch_panorama(self) -> Optional[Path]:
        panorama = self._panorama
        metadata = panorama.metadata
        logging.info(metadata)
        assert metadata
        with Image.new(
            "RGBA",
            (
                metadata.tileWidth * metadata.n_cols,
                metadata.tileHeight * metadata.n_rows,
            ),
        ) as image:
            tile_w, tile_h = metadata.tileWidth, metadata.tileHeight
            session = panorama.session

            def get_tile_image(tile: Tile):
                FileStorage.register(app_config.cache.base_root)
                tile_path = StreetViewTileImage(tile.url).path
                return (tile_path, tile)

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                future_to_url = {
                    executor.submit(get_tile_image, tile): tuple[Path, Tile]
                    for tile in panorama.tiles(session)
                }
                for future in concurrent.futures.as_completed(future_to_url):
                    try:
                        tile_path, tile = future.result()
                        assert tile_path
                        with Image.open(tile_path.as_posix()) as tile_image:
                            image.paste(
                                im=tile_image, box=(tile.x * tile_w, tile.y * tile_h)
                            )
                    except Exception as exc:
                        logging.exception(exc)
                    else:
                        pass
            full_path = (
                Path(app_config.cache.generated_root) / f"{self.filename}"
            )
            if max(image.width, image.height) > 4096:
                image.thumbnail((4096, 4096))
            image.save(full_path.as_posix())
            s3key = S3.upload(full_path, full_path.name)
            return full_path

    def __preprocess(
        self, img: Union[np.ndarray, Image.Image], is_cv2: bool = False
    ) -> torch.Tensor:
        """Preprocesses image"""
        if isinstance(img, np.ndarray) and is_cv2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if isinstance(img, Image.Image):
            # Sometimes images are RGBA
            img = img.convert("RGB")

        to_tensor = transforms.Compose([transforms.ToTensor()])
        img = to_tensor(img)
        assert len(img.shape) == 3, "input must be dim=3"
        assert img.shape[0] == 3, "input must be HWC"
        return img

    def __postprocess(
        self, img: torch.Tensor, to_cv2: bool = False
    ) -> Union[np.ndarray, Image.Image]:
        if to_cv2:
            img = np.asarray(img.to("cpu").numpy() * 255, dtype=np.uint8)
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        else:
            to_PIL = transforms.Compose([transforms.ToPILImage()])
            img = img.to("cpu")
            img = to_PIL(img)
            return img

    @property
    def unrolled(self) -> Path:
        # Rotation:
        rot = {
            "roll": 0,  #
            "pitch": 0,  # vertical
            "yaw": 0,  # horizontal
        }
        path = self.path
        w_face = 1920

        # Initialize equi2pers
        equi2cube = Equi2Cube(w_face=w_face, cube_format="horizon")
        device = torch.device("mps")

        # Open Image
        equi_img = Image.open(path)
        equi_img = self.__preprocess(equi_img).to(device)

        pers_img = equi2cube(equi_img, rots=rot)
        pers_img = self.__postprocess(pers_img)

        out_path = TempPath("output_equi2equi_torch_image.jpg")
        pers_img.save(out_path)
        return out_path
