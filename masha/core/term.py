from humanfriendly.terminal import (
    ansi_wrap,
    ANSI_COLOR_CODES,
)
from enum import Enum


class TermColor(Enum):
    BLACK = "black"
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"
    MAGENTA = "magenta"
    CYAN = "cyan"
    WHITE = "white"


def ccze(str, color: TermColor = None, bright=False):
    params = {}
    if color:
        params['color'] = ANSI_COLOR_CODES.get(color.value)
    if bright:
        params['bright'] = True
    if not len(params):
        return str
    return ansi_wrap(
        str,
        **params
    )
