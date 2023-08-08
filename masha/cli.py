import typer
from masha import (
    chat,
    code,
    image,
    knowledge_base,
    name, 
    text,
    hue
)

cli = typer.Typer()
cli.add_typer(chat.cli, name="chat")
cli.add_typer(code.cli, name="code")
cli.add_typer(image.cli, name="image")
cli.add_typer(knowledge_base.cli, name="kb")
cli.add_typer(name.cli, name="name")
cli.add_typer(text.cli, name="text")
cli.add_typer(hue.cli, name="hue")

@cli.command()
def serve():
    from masha.app import serve as app_serve
    from masha.firebase.db import OptionsDb
    from masha.image import OPTIONS as ImageOptions
    from masha.code.instruct import InstructOption
    OptionsDb().options(
        code={"instruct": InstructOption.values()},
        image=ImageOptions
    )    
    app_serve()

