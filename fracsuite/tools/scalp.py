import typer
from rich import print

from fracsuite.tools.specimen import Specimen

scalp_app = typer.Typer()

@scalp_app.command()
def sync():
    specimens = Specimen.get_all()

    for specimen in specimens:
        specimen.scalp.calc()
        specimen.scalp.save(specimen.scalp_dir)