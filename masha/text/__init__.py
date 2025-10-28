import logging
from pathlib import Path
from turtle import pd
from masha.core.term import TermColor, ccze
from masha.text.cover_letter import CoverLetter
from masha.text.detector import TextDetector
from masha.text.generator import TextGenerator
from masha.text.models import Payload
from masha.text.skills import SkillExtractor
import pycountry
from masha.text.genai import Gemini
from masha.text.openai import ChatGPT
from masha.core.request import make_response
from PIL import Image
from io import BytesIO
from numpy import asarray
from coreimage.terminal import print_term_image
from masha.core.image import download_image
import requests
import rich
from fastapi import APIRouter, Request, HTTPException
import typer
from typing_extensions import Annotated
from masha.text.config import text_config
import pandas as pd
import json

Gemini.register(text_config.genai)
TextGenerator.register(text_config.generator)
SkillExtractor.register(text_config.skills)
CoverLetter.register(text_config.cover_letter)
ChatGPT.register(text_config.openai)

router = APIRouter()
cli = typer.Typer()


@router.post("/generate")
async def api_generate(request: Request):
    message = await request.json()
    resp = TextGenerator.generateText(message["message"])
    return {"response": resp}


@router.post("/detect")
async def api_detect(request: Request):
    message = await request.json()
    code = TextDetector.detect(message["message"])
    lang = pycountry.languages.get(alpha_2=code)
    if not lang:
        raise HTTPException(404)
    return {"response": lang.name}


@router.post("/skills")
async def api_skills(request: Request):
    message = await request.json()
    tokens = SkillExtractor.getSkills(message["message"])
    return {"response": [t.model_dump(mode="json") for t in tokens]}


@router.post("/gemini")
async def api_gemini(payload: Payload):
    logging.info(payload)
    res = Gemini.ask(query=payload.message, source=payload.source)
    try:
        logging.debug(res.images)
        image_path = download_image(res.images[0])
        logging.debug(image_path)
        return make_response(image_path=image_path, message=res.content)
    except Exception:
        return make_response(message=res.content)


@cli.command()
def generate(prefix: Annotated[list[str], typer.Argument()]):
    print(TextGenerator.generateText(" ".join(prefix)))


@cli.command()
def detect(text: Annotated[list[str], typer.Argument()]):
    print(TextDetector.detect(" ".join(text)))


@cli.command()
def gemini(
    text: Annotated[list[str], typer.Argument()],
    file: Annotated[Path, typer.Option("-f", "--file")] = None,
):
    res = Gemini.ask(" ".join(text), source="localhost", fpath=file)
    if res.images:
        resp = requests.get(res.images[0])
        img = Image.open(BytesIO(resp.content))
        print_term_image(image=asarray(img), height=30)
    print(res.content)
    
@cli.command()
def chatgpt(
    text: Annotated[list[str], typer.Argument()]
):
    res = ChatGPT.ask(" ".join(text))
    print(res.output_text)
    
    
@cli.command()
def ask_json(
    text: Annotated[list[str], typer.Argument()],
    file: Annotated[Path, typer.Option("-f", "--file")] = None,
):
    if file:
        res = Gemini.ask_json(file.read_text())
    else:
        res = Gemini.ask_json(" ".join(text))
    if res.images:
        resp = requests.get(res.images[0])
        img = Image.open(BytesIO(resp.content))
        print_term_image(image=asarray(img), height=30)
    df = pd.DataFrame.from_dict(json.loads(res.content))
    with pd.option_context('display.max_colwidth', None):
        print(df)


@cli.command()
def skills(text: Annotated[list[str], typer.Argument()]):
    res = SkillExtractor.getSkills(" ".join(text))
    rich.print(res)


@cli.command("cover_letter")
def cli_cover(
    company: Annotated[str, typer.Option("-c", "--company")],
    name: Annotated[str, typer.Option("-n", "--name")] = "Alexander Spassov",
    job: Annotated[str, typer.Option("-j", "--job")] = "Full stack developer",
    background: Annotated[
        str, typer.Option("-b", "--background")
    ] = "Software engineer",
    experiences: Annotated[
        str, typer.Option("-e")
    ] = """Experienced software developer with solid system administration/automation
knowledge I am committed to maintaining code quality standards and best practices to
ensure robust and reliable software solutions. Through continuous learning and staying
abreast of emerging technologies, I seek to drive innovation and deliver exceptional
products to clients.""",
):
    prompt = f"coverletter name: {name} job: {job} at {company} background: {background} experiences: {experiences}"
    res = CoverLetter.generateText(prompt)
    print(ccze(res, color=TermColor.GREEN))
