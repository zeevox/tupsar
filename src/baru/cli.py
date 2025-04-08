import logging
from pathlib import Path
from typing import Final

import rich_click as click
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.logging import RichHandler

from baru.batch.chunk import ChunkConfig, create_batches
from baru.embedding.merge import merge_embeddings
from baru.embedding.submit import submit_batch_job
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
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
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
        path_type=Path,
    ),
    metavar="<dir>",
)
@click.argument(
    "file2",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
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
def diff(file1: Path, file2: Path, timeout: int, cleanup: int) -> None:
    """Embed the articles in the database."""

    def read_text(file: Path) -> str:
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


@cli.command(short_help="Split dataset for batch processing.")
@click.argument(
    "dataset",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
    metavar="<file>",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(
        exists=False,
        path_type=Path,
    ),
    default=None,
    help="Base output filename for prepared embedding jobs.",
)
@click.option(
    "-e",
    "--endpoint",
    type=str,
    default="/v1/embeddings",
    help="API endpoint to use.",
)
@click.option(
    "-m",
    "--model",
    type=str,
    default="text-embedding-3-large",
    help="Name of the model to use.",
)
def chunk(
    dataset: Path,
    endpoint: str,
    model: str,
    output: Path | None = None,
) -> None:
    """Prepare dataset for embedding."""
    create_batches(
        ChunkConfig(endpoint=endpoint, model_name=model),
        dataset,
        output,
    )


@cli.command(short_help="Submit a prepared embedding job.")
@click.argument(
    "job",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
    metavar="<file>",
)
def submit(job: Path) -> None:
    """Submit a prepared embedding job."""
    load_dotenv()  # Load API key from environment variables
    client = OpenAI()
    job_id = submit_batch_job(client, job)
    logger.info("Batch job submitted: %s", job_id)


@cli.command(short_help="Merge embeddings batch job output with the dataset.")
@click.argument(
    "dataset",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    metavar="<dataset>",
)
@click.argument(
    "jsonl",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    metavar="<batch_output>",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(exists=False, path_type=Path),
    default=None,
    help="Output filename for merged dataset.",
)
def merge(dataset: Path, jsonl: Path, output: Path | None = None) -> None:
    """Merge downloaded embeddings batch job output with a dataset."""
    if output is None:
        output = dataset.with_stem(f"{dataset.stem}_merged")

    merge_embeddings(dataset, jsonl, output)
    logger.info("Merged dataset saved to: %s", output)
