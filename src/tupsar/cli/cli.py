"""The `tupsar` command-line interface."""

import argparse
import asyncio
import logging
from logging import FileHandler
from pathlib import Path
from typing import Final

import asyncstdlib as a
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress
from rich_argparse import RichHelpFormatter

from tupsar.extractor.extractor import LangChainExtractor
from tupsar.extractor.pipeline import Model
from tupsar.file.path import unique_path

logger = logging.getLogger("tupsar")
LOG_LEVELS: Final[list[int]] = [
    logging.ERROR,
    logging.WARNING,
    logging.INFO,
    logging.DEBUG,
]


console: Console = Console()


def cli() -> None:
    """Handle the tupsar command-line interface."""
    if not load_dotenv():
        logger.warning("No .env file containing API keys found")

    # Set up argument parser
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter)
    parser.add_argument(
        "inputs",
        metavar="FILE",
        type=Path,
        nargs="+",
        help="Input file(s) to convert",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        metavar="DIR",
        type=Path,
        help="Output directory",
        default=Path("out"),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (can specify multiple times)",
    )
    parser.add_argument(
        "-e",
        "--extractor",
        choices=Model,
        nargs="+",
        default=Model.GEMINI_2_0,
        help="Extractor to use. Specify a second time to choose a fallback extractor.",
    )
    args = parser.parse_args()

    # Set logging verbosity
    log_level: int = min(args.verbose, len(LOG_LEVELS) - 1)
    logging.basicConfig(
        level=LOG_LEVELS[log_level],
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(rich_tracebacks=True, console=console),
            FileHandler("tupsar.log", encoding="utf-8"),
        ],
    )

    asyncio.run(
        main(
            args.inputs,
            args.output_path,
            LangChainExtractor(*args.extractor),
        )
    )


async def main(
    pages: list[Path], output_path: Path, extractor: LangChainExtractor
) -> None:
    """Run the main program entry-point."""
    output_path.mkdir(parents=True, exist_ok=True)

    with Progress(console=console) as progress:
        status = progress.add_task("Extracting articles...", total=len(pages))

        def update_progress(_: Path) -> None:
            progress.update(status, advance=1)

        extraction = extractor.extract_all(pages, callback=update_progress)
        async for counter, (path, article) in a.enumerate(extraction):
            output_file = unique_path(output_path / f"{path.stem}_{article.slug}.html")
            article.write_out(output_file)
            progress.update(status, description=f"Extracted {counter + 1} articles")
