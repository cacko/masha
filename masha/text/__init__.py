import logging
from masha.text.cover_letter import CoverLetter
from masha.text.detector import TextDetector
from masha.text.generator import TextGenerator
from masha.text.skills import SkillExtractor
import pycountry
from textacy.datasets import CapitolWords
from textacy.resources import DepecheMood
from textacy.lang_id import LangIdentifier
from masha.text.gemini import Gemini
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

Gemini.register(text_config.gemini)
TextGenerator.register(text_config.generator)
SkillExtractor.register(text_config.skills)
CoverLetter.register(text_config.cover_letter)



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
async def api_gemini(request: Request):
    message = await request.json()
    logging.info(message)
    res = Gemini.ask(message["message"])
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
def gemini(text: Annotated[list[str], typer.Argument()]):
    res = Gemini.ask(" ".join(text))
    if res.images:
        resp = requests.get(res.images[0])
        img = Image.open(BytesIO(resp.content))
        print_term_image(image=asarray(img), height=30)
    print(res.content)


@cli.command()
def skills(text: Annotated[list[str], typer.Argument()]):
    res = SkillExtractor.getSkills(' '.join(text))
    rich.print(res)


@cli.command("init-textacy")
def init_textacy():
    CapitolWords().download()
    DepecheMood().download()
    LangIdentifier(version=2.0).download()


# @bp.cli.command("cover_letter")
# @click.option("-n", "--name", default="Alexander Spasov")
# @click.option("-j", "--job", default="Full stack developer")
# @click.option("-c", "--company", default="Amazon AWS")
# @click.option(
#     "-b", "--background", default="Master of science in Information techologies"
# )
# @click.option(
#     "-e",
#     "--experiences",
#     default="I am the Author of Book and MTech in Machine Learning and achievement-driven professional with an experience of 3+ years in Data Science/Machine Learning/NLP/ Deep Learning/Data analytics. I am highly skilled in libraries like Sklearn, Numpy, Pandas, Matplotlib, Seaborn, Tensorflow, Faster-RCNN, Keras, Pytorch, FastAI, PowerBI/Tableau for Data Visualization, SQL/Oracle/NoSQL for databases and experience in NLP use cases related to named entity recognition, text summarization, text similarity, text generation",
# )
# def cli_cover(name: str, job: str, company: str, background: str, experiences: str):
#     prompt = f"coverletter name: {name} job: {job} at {company} background: {background} experiences: {experiences}"
#     res = CoverLetter.generateText(prompt)
#     print(ccze(res, color=TermColor.GREEN))
