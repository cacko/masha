from math import floor
from typing import Optional, Union
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
import numpy as np
from pathlib import Path
from corestring import truncate
from coreface import FaceColor
from corefile import TempPath
import requests
from uuid import uuid4
import shutil
from io import BytesIO


def load_remote_image(url):
    img_stream = BytesIO(requests.get(url).content)
    return cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)


def crop_offset(
    facial_area: list[int],
    w: int,
    h: int,
    distance: int = 5
) -> list[float]:
    left, top, right, bottom = facial_area
    offset = floor((right - left) / distance)
    return [max(0, left - offset),
            max(0, top - offset),
            min(right + offset, w),
            min(bottom + offset, h)]


def add_tags(
    img_path,
    faces: list[tuple[Optional[list[int]], str]]
) -> Image.Image:
    pil_image = Image.open(img_path)
    h, w = pil_image.height, pil_image.width
    fnt_path = Path(__file__).parent / "MesloLGS NF Regular.ttf"
    for facial_area, name in faces:
        draw = ImageDraw.Draw(pil_image)
        if facial_area:
            left, top, right, bottom = crop_offset(facial_area, w, h)
            draw.rectangle(((left, top), (right, bottom)),
                           outline=(239, 108, 0))
            if name:
                fnt_size = max(
                    10, min(25, floor((bottom - top) / 20))
                )
                fnt = ImageFont.truetype(fnt_path.as_posix(), fnt_size)
                name = f"{name}"
                text_width, text_height = draw.textsize(name, font=fnt)
                if right - left < text_width + 6:
                    name = truncate(name)
                draw.rectangle((
                    (left, bottom - text_height - 10), (right, bottom)
                ), fill=(239, 108, 0), outline=(239, 108, 0))
                draw.text((left + 6, bottom - text_height - 5),
                          name, fill=(255, 243, 224), font=fnt)

        del draw
    return pil_image


def resize(img_path: Path, max_size: int = 500):
    img = Image.open(img_path.as_posix())
    img.thumbnail((max_size, max_size))
    img.save(img_path.as_posix())


def show_tagged(img_path: Path, names: str = "", wait=10):
    img = cv2.imread(img_path.as_posix())
    window_name = names
    cv2.imshow(window_name, img)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(wait * 1000)
    cv2.destroyAllWindows()


def show_image(image=None, path: Optional[Path] = None):
    if path:
        image = cv2.imread(path.as_posix())
    cv2.imshow("unknown", image)
    cv2.setWindowProperty("unknown", cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(10 * 1000)



def normalize_image(photo: Path) -> Path:
    img = cv2.imread(photo.as_posix())
    image_rows, image_cols = img.shape[:2]
    row_ratio = 2000 / float(image_rows)
    col_ratio = 2000 / float(image_cols)
    ratio = min(row_ratio, col_ratio)
    img = cv2.resize(
        img,
        dsize=(int(2000 * row_ratio), int(2000 * col_ratio)),
        fx=int(ratio),
        fy=int(ratio),
        interpolation=cv2.INTER_LANCZOS4
    )
    tmp_path = TempPath(f"{photo.stem}.png")
    cv2.imwrite(tmp_path.as_posix(), img)
    return tmp_path


def load_image(img: Union[str, np.ndarray], color: Optional[FaceColor] = None):
    if type(img).__module__ == np.__name__:
        img_array = img
    elif isinstance(img, str):
        img_array = cv2.imread(img)
    else:
        raise NotImplementedError
    match color:
        case FaceColor.GREYSCALE:
            if isinstance(img_array, np.ndarray):
                res = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
                return res
        case _:
            return img_array
        


def download_image(url: str) -> TempPath:
    tmp_file = TempPath(f"{uuid4()}.jpg")
    response = requests.get(url, stream=True)
    with tmp_file.open("wb") as out_file:
        shutil.copyfileobj(response.raw, out_file)
    return tmp_file
