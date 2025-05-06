from masha.text.genai import Gemini
from typing import Optional
from corefile import TempPath
from pydantic import BaseModel
import typer
from pathlib import Path
from typing_extensions import Annotated
from masha.image.age import AgeClassifier
from masha.image.classify import Age, Attraction, Classifier, Ethnic, Gender
from masha.image.cli import cli
from masha.image.deepface import AgeClient, RaceClient
from masha.image.router import router
from rich import print
from fastapi import UploadFile, File
from masha.core.request import uploaded_file
from coreimage.terminal import print_term_image
from coreimage.transform.crop import Cropper


@cli.command()
def ask(
    img_path: Annotated[Path, typer.Argument()],
    query: Annotated[list[str], typer.Argument()],
):
    print_term_image(image_path=img_path, height=20)
    res = Gemini.ask_image(img_path, query)
