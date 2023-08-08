import logging
from cachable import BinaryStruct
from cachable.storage.filestorage.image import CachableFileImage as CachableFile
from cachable.storage.file import FileStorage
from masha.knowledge_base.summarization import WikiSummarization
from masha.knowledge_base.wiki import Wiki
from masha.knowledge_base.ask import Ask
from masha.knowledge_base.wolfram import Wolfram
from rich import print
from masha.core.request import make_multipart_response
from PIL import Image
from svglib.svglib import svg2rlg
from cachable.request import Request
from reportlab.graphics import renderPM
from opengraph_parse import parse_page
from masha.chat import Sarcastic
from corestring import string_hash
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, File, UploadFile, Form
import typer
from typing_extensions import Annotated

router = APIRouter()
cli = typer.Typer()

def saracstic_reponse(question: str):
    resp = Sarcastic.sarcastic(
        message=question,
    )
    return {"response": resp}


class KbImage(CachableFile):

    _filename: Optional[str] = None

    def __init__(self, url: str, query: str):
        self._url = url
        self._query = query

    @property
    def storage(self):
        return FileStorage

    @property
    def filename(self):
        if not self._filename:
            self._filename = f"{string_hash(self._query)}.png"
        return self._filename

    @property
    def url(self):
        return self._url

    def _init(self):
        if self.isCached:
            return
        try:
            req = Request(self.url)
            content = req.binary
            content_type = content.type

            if "text/html" in content_type:
                tags = parse_page(self.url)
                img_url = tags.get("og:image")  # type: ignore
                if img_url:
                    self._url = img_url
                    return self._init()
            assert self._path
            self._path.write_bytes(content.binary)
            if "image/svg+xml" in content_type:
                drawing = svg2rlg(self._path.as_posix())
                renderPM.drawToFile(drawing, self._path.as_posix(), fmt="PNG")
            img = Image.open(self._path.as_posix())
            rt = 500 / img.width
            img = img.resize((int(img.width * rt), int(img.height * rt)))
            img.save(self._path.as_posix(), "PNG")
            self.tocache(BinaryStruct(
                binary=self._path.read_bytes(),
                type="image/png"
            ))
        except Exception:
            self._path = self.DEFAULT


@router.post("/ask")
async def route_ask(
    request: Request
):
    try:
        question = None
        if message := await request.json():
            question = message.get("message")
        if not question:
            return saracstic_reponse("I am stupid")
        resp = Ask.ask(question)
        if not resp:
            return saracstic_reponse(question)
        text = resp.message
        image_url = resp.image_url
        if all([not text, not image_url]):
            return saracstic_reponse(question)
        if image_url:
            kbimage = KbImage(f"{image_url}", question)
            assert kbimage.path
            return make_multipart_response(kbimage.path, text if text else "")
        return {"response": text}
    except AssertionError:
        return saracstic_reponse("I am stupid")


@router.post("/tell")
async def tell_command(
    request: Request
):
    try:
        message = await request.json()
        question = message.get("message")
        if not question:
            return saracstic_reponse(question)
        wiki = Wiki(question)
        summary = wiki.summary
        if not summary:
            return saracstic_reponse(question)
        resp = WikiSummarization.summarization(summary)
        return {"response": resp}
    except (FileNotFoundError, AttributeError):
        return saracstic_reponse("I am stupid")


@cli.command()
def wolfram(
    question: Annotated[list[str], typer.Argument()],
):
    resp = Wolfram.query(' '.join(question))
    if not resp:
        return
    if resp.image_url:
        kb_image = KbImage(f"{resp.image_url}", ' '.join(question))
        resp.image_url = kb_image.path
        if resp.image_url:
            resp.image_url = resp.image_url.as_posix()
    text, image = resp.message, resp.image_url
    print(text)
    if image:
        Image.open(image).show()


@router.post("/wtf")
async def wtf_command(
    request: Request
):
    try:
        message = await request.json()
        question = message.get("message")
        if not question:
            return saracstic_reponse("I am stupid")
        resp = Wolfram.query(question)
        if not resp:
            return saracstic_reponse("I am stupid")
        text = resp.message
        image_path = resp.image_url
        if all([not text, not image_path]):
            return saracstic_reponse("I am stupid")
        if image_path:
            if not text:
                text = ""
            kbimage = KbImage(f"{image_path}", question)
            assert kbimage.path
            return make_multipart_response(kbimage.path, text)
        elif text:
            return {"response": text}
        return saracstic_reponse("I am stupid")
    except (FileNotFoundError, AttributeError):
        return saracstic_reponse("I am stupid")


@cli.command()
def tell(
    question: Annotated[list[str], typer.Argument()],
):
    wiki = Wiki(' '.join(question))
    summary = wiki.summary
    print(WikiSummarization.summarization(summary))


@cli.command()
def sum(
    question: Annotated[list[str], typer.Argument()],
):
    print(WikiSummarization.summarization(' '.join(question)))


@cli.command()
def ask(
    question: Annotated[list[str], typer.Argument()],
):
    try:
        resp = Ask.ask(' '.join(question))
        assert resp
        if resp.image_url:
            resp.image_url = KbImage(f"{resp.image_url}", question).path
            if resp.image_url:
                resp.image_url = resp.image_url.as_posix()
        text, image = resp.message, resp.image_url
        print(text)
        if image:
            Image.open(image).show()
    except AssertionError as e:
        logging.exception(e)
