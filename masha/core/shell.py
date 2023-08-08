from subprocess import PIPE, Popen, STDOUT, call
import os
from typing import Generator, Optional
import shlex
import logging


class ShellMeta(type):
    def __call__(cls, *args, **kwargs):
        return type.__call__(cls, *args, **kwargs)

    def output(cls, *args) -> Generator[str, None, None]:
        yield from cls(*args).getOutput()


class Shell(object, metaclass=ShellMeta):
    executable: Optional[str] = None
    executable_arguments: list[str] = []

    OUTPUT_IGNORED: list[str] = []
    OUTPUT_REPLACED: list[str] = []

    def __init__(self, args: str):
        self.args = shlex.split(args)
        self.__post_init__()

    def __post_init__(self):
        pass

    @property
    def environment(self):
        return dict(
            os.environ
        )

    def getOutput(self) -> Generator[str, None, None]:
        assert self.executable
        assert self.args
        cmd = (self.executable, *self.executable_arguments, *self.args)
        logging.info(f"Shell.getOutput: {cmd}")
        with Popen(
            cmd,
            stdout=PIPE,
            stderr=STDOUT,
            env=self.environment,
        ) as p:
            assert p.stdout
            for b_line in iter(p.stdout.readline, b""):
                line = b_line.decode().strip()
                if any([x in line for x in self.OUTPUT_IGNORED]):
                    continue
                if prefix := next(
                    filter(lambda x: line.startswith(x), self.OUTPUT_REPLACED), None
                ):
                    line = line.removeprefix(prefix).strip()
                    line = f"{line.upper()}\n"
                yield line
            if p.returncode:
                return None

    def execute(self):
        cmd = [self.executable, *self.executable_arguments, *self.args]
        logging.info(f"Shell.execute: {shlex.join(cmd)}")
        return call(shlex.join(cmd), shell=True, env=self.environment)
