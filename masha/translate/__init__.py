from masha.translate.translator import Translator
from fastapi import APIRouter, Request, HTTPException, Path
import typer
from typing_extensions import Annotated

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
    "sq_en",
    "en_sq",
    "en_ur",
    "ur_en",
    "pa_en",
]


@router.post("/{lang}")
async def translate_command(request: Request, lang: Annotated[str, Path(title="lang")]):
    if lang not in SUPPORTED:
        raise HTTPException(404)
    message = await request.json()
    assert message
    t = Translator.translated(lang, message["message"])
    return {"response": t}
