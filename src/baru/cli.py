import logging
import pathlib
from pathlib import Path
from typing import Final

import rich_click as click
from rich.console import Console
from rich.logging import RichHandler

from baru.evaluation.diff import compute_diff, diff_to_rich
from tupsar.model.article import Article

LOG_LEVELS: Final[list[int]] = [
    logging.ERROR,
    logging.WARNING,
    logging.INFO,
    logging.DEBUG,
]
CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
CLI_NAME: Final[str] = "baru"

console: Console = Console()
logger = logging.getLogger(CLI_NAME)


def configure_logging(_ctx: click.Context, _param: click.Parameter, value: int) -> None:
    """Set log level with click callback."""
    logging.basicConfig(
        level=LOG_LEVELS[value],
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(rich_tracebacks=True, console=console),
            logging.FileHandler(Path(CLI_NAME).with_suffix(".log"), encoding="utf-8"),
        ],
    )


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-v",
    "--verbose",
    count=True,
    type=click.IntRange(0, len(LOG_LEVELS), max_open=True, clamp=True),
    metavar="",
    help="Increase verbosity (can specify multiple times)",
    is_eager=True,  # process option before others
    expose_value=False,
    callback=configure_logging,
)
def cli() -> None:
    """Handle the baru command-line interface."""


@cli.command(short_help="Compare different article extractions.")  # @cli, not @click!
@click.argument(
    "file1",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=pathlib.Path,
    ),
    metavar="<dir>",
)
@click.argument(
    "file2",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=pathlib.Path,
    ),
    metavar="<dir>",
)
@click.option(
    "-c",
    "--cleanup",
    metavar="<edit cost>",
    type=click.IntRange(0, clamp=True),
    default=8,
    help="Efficiency cleanup edit cost. Increase computational efficiency by "
    "factoring out short commonalities which are not worth the overhead. "
    "The larger the edit cost, the more aggressive the cleanup.",
)
@click.option(
    "-t",
    "--timeout",
    type=click.IntRange(0, clamp=True),
    metavar="<seconds>",
    default=0,
    help="If the mapping phase of the diff computation takes longer than "
    "this, then the computation is truncated and the best solution to "
    "date is returned. While guaranteed to be correct, it may not be "
    "optimal. A timeout of '0' allows for unlimited computation.",
)
def diff(file1: pathlib.Path, file2: pathlib.Path, timeout: int, cleanup: int) -> None:
    """Embed the articles in the database."""

    def read_text(file: pathlib.Path) -> str:
        """Read text from a file."""
        if file.suffix == ".html":
            return Article.read_in(file).txt
        return file.read_text()

    diffs = compute_diff(
        read_text(file1),
        read_text(file2),
        edit_cost=cleanup,
        timeout=timeout,
    )
    rich_diff = diff_to_rich(diffs)
    console.print(rich_diff)
