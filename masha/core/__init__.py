import time
from contextlib import contextmanager
import logging
from typing import Optional
import psutil
from appdirs import user_data_dir
import os
from pathlib import Path


data_dir = Path(user_data_dir("masha"))
if not data_dir.exists():
    data_dir.mkdir(exist_ok=True)

pid_file = data_dir / "masha.pid"


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_available_memory() -> int:
    return psutil.virtual_memory().available


@contextmanager
def perftime(name, silent=False):
    st = time.perf_counter()
    try:
        yield
    finally:
        if not silent:
            total = time.perf_counter() - st
            logging.info(f"{name} -> {total}s")


def check_pid() -> Optional[int]:
    try:
        assert pid_file.exists()
        pid = pid_file.read_text()
        assert pid
        os.kill(int(pid), 0)
        return int(pid)
    except (AssertionError, ValueError, OSError):
        return None


class Choices(object):
    @classmethod
    def values(cls):
        return [m.value for m in cls.__members__.values()]

    @classmethod
    def keys(cls):
        return [m.lower() for m in cls.__members__.keys()]

    @classmethod
    def members(cls):
        return cls.__members__.values()
