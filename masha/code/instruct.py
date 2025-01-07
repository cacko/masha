from enum import StrEnum
import logging
from typing import Optional
from tifffile import format_size
import torch
from masha.code.config import InstructConfig
from masha.core import Choices, perftime
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, set_seed
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
            self.__model = AutoModelForCausalLM.from_pretrained(
                model_path.as_posix(), torch_dtype=torch.float16, trust_remote_code=True
            ).to(TORCH_DEVICE)
        return self.__model

    @property
    def tokenizer(self):
        if not self.__tokenizer:
            model_path = DATA_ROOT / self.__modelId
            self.__tokenizer = AutoTokenizer.from_pretrained(
                model_path.as_posix(), trust_remote_code=True
            )
        return self.__tokenizer

    @property
    def prompt_system(self):
        return ""

    def get_prompt(self, user_query: str):
        result = [f"<s>[INST] "]
        try:
            assert self.prompt_system
            result.append(f"<<SYS>\\n{self.prompt_system}\\n<</SYS>>\\n\\n")
        except AssertionError:
            pass
        result.append(user_query.strip())
        result.append(f"[/INST]")
        return "".join(result)

    def get_result(self, user_query):
        with perftime(f"{self.__class__.__name__}"):
            prompt = self.get_prompt(user_query)
            tokenizer = self.tokenizer
            model: LlamaForCausalLM = self.model
            logging.debug(prompt)
            input_ids = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            )["input_ids"]
            generated_ids = model.generate(
                input_ids.to(TORCH_DEVICE),
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
            )
            output = generated_ids[:, input_ids.shape[1] :].to(TORCH_DEVICE)
            result = tokenizer.batch_decode(output, skip_special_tokens=True)
            return self.clean_result("".join(result))

    def clean_result(self, result: str):
        return result.strip().replace("```\n```", "```")


class InstructGeneral(Instruct):
    option = InstructOption.GENERAL

    def get_result(self, user_query):
        with perftime(f"{self.__class__.__name__}"):
            prompt = self.get_prompt(user_query)
            tokenizer = self.tokenizer
            model: LlamaForCausalLM = self.model
            logging.debug(prompt)
            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"]
            set_seed(randint(40, 100))
            generated_ids = model.generate(
                input_ids.to(TORCH_DEVICE),
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
                top_k=10,  # default=10
                # top_p=0.5, # default=0.9
                temperature=0.6,  # default=0.
                do_sample=True,
                num_return_sequences=1,
            )
            output = generated_ids[:, input_ids.shape[1] :].to(TORCH_DEVICE)
            result = tokenizer.batch_decode(output, skip_special_tokens=True)
            return self.clean_result("".join(result))


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
