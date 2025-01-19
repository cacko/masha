from enum import StrEnum
from masha.translate.translator import Translator
from fastapi import APIRouter, Request, HTTPException, Path as FPath
import typer
from pathlib import Path
from typing_extensions import Annotated
from transformers import MarianMTModel
import torch

router = APIRouter()
cli = typer.Typer()

SUPPORTED = [
    "en_es",
    "es_en",
    "bg_en",
    "en_bg",
    "en_cs",
    "cs_en",
    "pl_en",
    "en_pl",
    "en_fr",
    "fr_en",
    "en_it",
    "it_en",
]

lang_options = StrEnum("langs", SUPPORTED)


@router.post("/{lang}")
async def translate_command(
    request: Request, lang: Annotated[str, FPath(title="lang")]
):
    if lang not in SUPPORTED:
        raise HTTPException(404)
    message = await request.json()
    assert message
    t = Translator.translated(lang, message["message"])
    return {"response": t}


@cli.command()
def translate(
    text: Annotated[list[str], typer.Argument()],
    langs: Annotated[lang_options, typer.Option("-l", "--langs")],
):
    assert text
    t = Translator.translated(langs, " ".join(text))
    print(t)
