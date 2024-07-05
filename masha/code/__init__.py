from masha.code.instruct import (
    Instruct,
    InstructGeneral,
    InstructJavascript,
    InstructOption,
    InstructPHP,
    InstructPython,
)
from masha.code.summarization import CodeSummarization
from masha.code.config import code_config
import click
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException
from fastapi import Path as FastPath
import typer
from typing_extensions import Annotated

CodeSummarization.register(code_config.summarization)
InstructGeneral.register(code_config.instruct)
InstructPython.register(code_config.instruct_python)
InstructPHP.register(code_config.instruct_php)
InstructJavascript.register(code_config.instruct_javascript)

router = APIRouter()
cli = typer.Typer()


@cli.command()
@click.argument("patch_file")
def commit(patch_file: Annotated[Path, typer.Argument()]):
    content = patch_file.read_text()
    print(CodeSummarization.summarization(content))


@router.put("/diff")
async def route_tag(request: Request):
    diff = await request.body()
    try:
        msg = CodeSummarization.summarization(diff.decode(errors='replace'))
        return {"message": msg}
    except AttributeError:
        pass
    raise HTTPException(404)


@cli.command()
def instruct(
    query: Annotated[list[str], typer.Argument()],
    model: Annotated[
        InstructOption, typer.Option("-m", "--models")
    ] = InstructOption.GENERAL,
):
    user_prompt = " ".join(query)
    cls = Instruct.cls_for_option(option=model)
    result = cls.instruct(user_prompt)
    print(result)


@router.post("/instruct/{model}")
async def route_instruct(
    request: Request, model: Annotated[str, FastPath(title="model")]
):
    data = await request.json()
    try:
        query = data.get("message")
        assert query
        cls = Instruct.cls_for_option(option=InstructOption(model))
        assert cls
        msg = cls.instruct(query)
        assert msg
        return {"response": msg}
    except AttributeError:
        pass
    raise HTTPException(404)
