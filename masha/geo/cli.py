from typing import Annotated
import typer
from .places import sample_get_place

cli = typer.Typer()


@cli.command()
def place(
    query: Annotated[list[str], typer.Argument()],
):
    res = sample_get_place(' '.join(query))
