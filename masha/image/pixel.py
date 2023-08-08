from pathlib import Path
from pixelme import Pixelate
from typing import Optional


def pixelate(
    img_path: Path,
    padding=480,
    grid_lines=True,
    block_size=8,
    result_path: Optional[Path] = None
):

    block_size = max(int(padding / block_size), 8)

    pix = Pixelate(
        img_path,
        padding=padding,
        grid_lines=grid_lines,
        block_size=block_size,
        result_path=result_path
    )
    res = pix.image_path
    return res
