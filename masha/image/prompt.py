import logging
from pydantic import BaseModel
import torch
from pathlib import Path
from os import environ
from transformers import (
    pipeline,
    AutoModelForCausalLM,                                                                                                                     
    AutoTokenizer,
)
from .config import image_config
from functools import reduce
import re
import rich


class Config(BaseModel):
    model: str
    tokenizer: str
    option: str


class PromptType(type):
    configs: dict[str, Config] = {}
    classes: dict[str, "PromptType"] = {}
    IGNORED_ARGS = ["w", "h"]

    def __call__(cls):
        return type.__call__(
            cls,
            cls.tokenizerPath,
            cls.modelPath
        )

    def register(cls, model: str, tokenizer: str):
        cls.configs[cls.__name__] = Config(
            model=model, tokenizer=tokenizer, option=cls.__name__[-1]
        )
        cls.classes[cls.__name__] = cls

    @property
    def dataRoot(cls) -> Path:
        return Path(environ.get("HUGGINGROOT", "."))

    @property
    def modelPath(cls) -> Path:
        return cls.dataRoot / cls.configs[cls.__name__].model

    @property
    def tokenizerPath(cls) -> Path:
        return cls.dataRoot / cls.configs[cls.__name__].tokenizer

    @property
    def device(cls):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def generate(cls, input: str) -> str:
        return cls().do_prompt(input)

    @property
    def options(cls) -> list[str]:
        return [x.option for x in cls.configs.values()]

    def cls_for_option(cls, opt: str) -> "PromptType":
        res = next(
            filter(lambda x: cls.configs[x].option == opt, cls.configs.keys()), None
        )
        assert res
        return cls.classes[res]


class Prompt(object, metaclass=PromptType):
    num_return_sequences = 1

    def __init__(self, model_path: Path, tokenizer_path: Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path.as_posix())
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model = AutoModelForCausalLM.from_pretrained(model_path.as_posix()).to(
            Prompt.device
        )  # type: ignore

    def do_prompt(self, input: str) -> str:
        model = self.model
        tokenizer = self.tokenizer
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        outputs = generator(
            f"{input}",
            num_return_sequences=self.num_return_sequences,
            return_full_text=False,
        )
        result = " ".join(
            [
                input,
                " ".join([x.get("generated_text", "") for x in outputs]),  # type: ignore
            ]
        )
        rich.print(result)
        result = self._normalize(result)
        logging.info(f"AUTO-PROMPT {self.__class__.modelPath.stem}: {result}")
        return result

    def _normalize(self, input: str) -> str:
        args = re.findall(r"\s([-\u2014]+[^ ]+\s[^ ]+)", input)
        for sf in args:
            input = input.replace(sf, "")
        input = re.sub(r"(\s*,\s*|\n)", r",", input)

        input_list: list[str] = reduce(
            lambda r, x: [*r, *([x] if x not in r else [])], input.split(" "), []
        )

        args_dict = dict(map(lambda x: x.strip("-\u2014").split(" "), args))

        final_args = [
            f"--{k} {v}" for k, v in args_dict.items() if k not in Prompt.IGNORED_ARGS
        ]

        return " ".join(input_list + final_args)


def register_prompts():
    for pr in image_config.prompts:
        cls = pr.name
        inst = type(cls, (Prompt,), {})
        inst.register(pr.model, pr.tokenizer)