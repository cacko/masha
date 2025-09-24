from enum import StrEnum
import logging
from typing import Optional
import torch
from humanfriendly import format_size
from masha.code.config import InstructConfig
from masha.core import Choices, perftime
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from masha.pipelines import DATA_ROOT, TORCH_DEVICE
from torch.mps import (
    empty_cache,
    current_allocated_memory,
)
import gc
from random import randint


class InstructOption(Choices, StrEnum):
    PHP = "php"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    GENERAL = "general"


class InstructMeta(type):
    _instance: Optional["Instruct"] = None
    configs: dict[InstructOption, InstructConfig] = {}
    clss: dict[InstructOption, "Instruct"] = {}

    def __call__(cls, *args, **kwds):
        if not cls._instance:
            assert cls.option in cls.configs
            cls._instance = type.__call__(
                cls, cls.configs[cls.option].model, *args, **kwds
            )
        return cls._instance

    def register(cls, config: InstructConfig):
        cls.configs[cls.option] = config
        cls.clss[cls.option] = cls

    def cls_for_option(cls, option: InstructOption):
        return cls.clss[option]

    def instruct(cls, text) -> str:
        worker = cls()
        result = worker.get_result(user_query=text)
        worker = None
        cls._instance = None
        logging.info(f"Memory allocated - {format_size(current_allocated_memory())}")
        return result


class Instruct(object, metaclass=InstructMeta):

    __model: Optional[AutoModelForCausalLM] = None
    __tokenizer: Optional[AutoTokenizer] = None

    def __init__(self, model: str) -> None:
        self.__modelId = model

    def __del__(self):
        logging.info("Releasing cache")
        self.pipeline = None
        self.params = None
        gc.collect()
        empty_cache()

    @property
    def model(self):
        if not self.__model:
            model_path = DATA_ROOT / self.__modelId
            if "/" in self.__modelId:
                model_path = self.__modelId
            self.__model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, trust_remote_code=True
            ).to(TORCH_DEVICE)
        return self.__model

    @property
    def tokenizer(self):
        if not self.__tokenizer:
            model_path = DATA_ROOT / self.__modelId
            if "/" in self.__modelId:
                model_path = self.__modelId
            self.__tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
        return self.__tokenizer

    @property
    def prompt_system(self):
        return "You are a code completion assistant."

    def get_result(self, user_query: str):
        with perftime(f"{self.__class__.__name__}"):
            tokenizer = self.tokenizer
            model = self.model
            messages = [
                {"role": "system", "content": self.prompt_system},
                {"role": "user", "content": user_query.strip()},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(TORCH_DEVICE)

            eos_token_ids = [151659, 151661, 151662, 151663, 151664, 151643, 151645]
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=eos_token_ids,
            )[0]
            # The generated_ids include prompt_ids, we only need to decode the tokens after prompt_ids.
            output_text = tokenizer.decode(
                generated_ids[len(model_inputs.input_ids[0]) :],
                skip_special_tokens=True,
            )
            return output_text

    def clean_result(self, result: str):
        return result.strip().replace("```\n```", "```")


class InstructGeneral(Instruct):
    option = InstructOption.GENERAL


class InstructPython(Instruct):
    option = InstructOption.PYTHON

    @property
    def prompt_system(self):
        return "Provide answers in Python"


class InstructPHP(Instruct):

    option = InstructOption.PHP

    @property
    def prompt_system(self):
        return "Provide answers in PHP"


class InstructJavascript(Instruct):

    option = InstructOption.JAVASCRIPT

    @property
    def prompt_system(self):
        return "Provide answers in Javascript"
