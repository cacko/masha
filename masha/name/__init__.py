from masha.name.demographics import Demographics
from fastapi import APIRouter, Request, HTTPException, File, UploadFile, Form
import typer
from typing_extensions import Annotated
from rich import print

router = APIRouter()
cli = typer.Typer()


@router.post("/gender")
async def api_gender(request: Request):
    message = await request.json()
    gender = Demographics.gender(message["message"])
    return {"response": gender.value}


@router.post("/race")
async def api_race(request: Request):
    message = await request.json()
    race = Demographics.race(message["message"])
    return {"response": race.value}


@cli.command()
def race(
    name: Annotated[list[str], typer.Argument()],
):
    print(Demographics.race(" ".join(name)).value)
