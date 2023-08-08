import logging
import click
import torch
from masha.config import ChatConfig
from masha.pipelines import TORCH_DEVICE
from masha.pipelines.conversational import Conversational
from masha.translate.translator import Translator
from masha.text.detector import TextDetector
from humanfriendly.prompts import prompt_for_input
from typing import Any, Optional
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)
from uuid import uuid4
from fastapi import APIRouter, Request, HTTPException
import typer
from typing_extensions import Annotated

router = APIRouter()
cli = typer.Typer()

# bp = Blueprint("chat", __name__, url_prefix="/chat")

NATIVE_LANGUAGE = "en"
TRANSLATIONS = ["bg", "es", "pl", "cs", "sq"]


def output(txt: str, color="bright_blue"):
    click.secho(txt, fg=color)


class ChatDialogMeta(type):
    config: ChatConfig

    def __call__(cls, model_name: str, source: str, *args, **kwds):
        return type.__call__(cls, model_name, *args, **kwds)

    @classmethod
    def register(cls, config: ChatConfig):
        cls.config = config

    def get_instance(cls) -> "ChatDialog":
        return cls(cls.config.dialog_model, cls.config.dialog_tokenizer)

    def dialog(cls, source: str = "", message: str = "", lang=None, **kw) -> str:
        return cls(
            cls.config.dialog_model, cls.config.dialog_tokenizer
        ).conversation_response(text=message, out_lang=lang, **kw)

    def phrase(cls, source: str = "", message: str = "", lang=None, **kw) -> str:
        return cls(
            cls.config.dialog_phrase_model,
            cls.config.dialog_phrase_tokenizer,
        ).conversation_response(text=message, out_lang=lang, **kw)

    def sarcastic(cls, source: str = "", message: str = "", lang=None, **kw) -> str:
        return cls(
            cls.config.dialog_sarcastic_model,
            cls.config.dialog_sarcastic_tokenizer,
        ).conversation_response(text=message, out_lang=lang, **kw)


class ChatTranslator(object):

    __native_input: Optional[str] = None
    __input_lang: Optional[str] = None

    def __init__(self, text, out_lang) -> None:
        self.__input = text
        self.__output_lang = out_lang

    @property
    def input_lang(self) -> str:
        if not self.__input_lang:
            self.__input_lang = TextDetector.detect(self.__input)
        return self.__input_lang

    @property
    def output_lang(self) -> str:
        if not self.__output_lang:
            assert self.__input_lang
            self.__output_lang = self.__input_lang
        return self.__output_lang

    @property
    def native_input(self) -> str:
        try:
            if not self.__native_input:
                if self.input_lang in TRANSLATIONS:
                    native_text = Translator.translated(
                        lang=f"{self.input_lang.lower()}_{NATIVE_LANGUAGE}",
                        text=self.__input,
                    )
                    if not native_text:
                        native_text = self.__input
                    self.__native_input = native_text
                    logging.info(f"{self.__input} -> {self.__native_input}")
            assert self.__native_input
            return self.__native_input
        except AssertionError:
            return self.__input


class ChatDialog(Conversational, metaclass=ChatDialogMeta):

    def __init__(self, model_name, tokenizer_path=None):
        self.__pipeline = None
        super().__init__(model_name, tokenizer_path)

    @property
    def pipeline(self):
        if not self.__pipeline:
            self.__pipeline = pipeline(
                "text-generation",
                model=self.modelPath.as_posix(),
                model_kwargs={
                    "torch_dtype": torch.float16,
                    "low_cpu_mem_usage": True,
                },
                device=TORCH_DEVICE,
            )
        return self.__pipeline

    def conversation_response(self, text: str, **kwargs) -> str:
        resp = ""
        out_lang = kwargs.get("out_lang", "en")
        pipe = self.pipeline
        messages = [
            {
                "role": "system",
                "content": "You always responds in sarcastic manner!",
            },
        ]
        if kwargs and kwargs.get("detect_lang"):
            trans = ChatTranslator(text, out_lang)
            native_text = trans.native_input
            messages.append(
                {"role": "user", "content": native_text},
            )
            out_lang = trans.output_lang
        else:
            messages.append({"role": "user", "content": text})
        prompt = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = pipe(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        resp = outputs[0]["generated_text"][len(prompt) :]

        if out_lang and out_lang in TRANSLATIONS:
            resp = Translator.translated(
                lang=f"{NATIVE_LANGUAGE}_{out_lang.lower()}", text=resp
            )
        return resp


class Sarcastic(ChatDialog):

    @property
    def model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.modelPath)

    @property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.modelPath, padding_side="left")


class Phrase(ChatDialog):

    @property
    def model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.modelPath)

    @property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.modelPath)


@router.post("/dialog")
async def api_dialog(request: Request):
    data = await request.json()
    resp = ChatDialog.dialog(**data)  # type: ignore
    return {"response": resp}


@router.post("/phrase")
async def api_phrase(request: Request):
    data = await request.json()
    resp = Phrase.phrase(**data)  # type: ignore
    return {"response": resp}


@router.post("/sarcastic")
async def api_sarcastic(request: Request):
    data = await request.json()
    resp = Sarcastic.sarcastic(**data)  # type: ignore
    return {"response": resp}


@cli.command()
def dialog():
    try:
        src = uuid4()
        chat = ChatDialog.get_instance()
        while True:
            tt = prompt_for_input("me > ")
            print(f"bot> {chat.conversation_response(text=tt,detect_lang=True)}")
    except KeyboardInterrupt:
        pass


@cli.command()
def sarcastic():
    try:
        src = uuid4()
        while True:
            tt = prompt_for_input("me > ")
            print(
                f"bot> {Sarcastic.sarcastic(source=src,message=tt, detect_lang=True)}"
            )
    except KeyboardInterrupt:
        pass


@cli.command()
def phrase(phrase: Annotated[str, typer.Argument()]):
    resp = Phrase.phrase(source=uuid4(), message=phrase)  # type: ignore
    output(resp)
