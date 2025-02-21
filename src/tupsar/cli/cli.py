"""The `tupsar` command-line interface."""

import argparse
import asyncio
import logging
from logging import FileHandler
from pathlib import Path

import asyncstdlib as a
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich_argparse import RichHelpFormatter

from tupsar.extractor import BaseExtractor
from tupsar.extractor.langchain import LangChainExtractor
from tupsar.file.path import unique_path

logger = logging.getLogger("tupsar")


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
        choices=LangChainExtractor.Model,
        default="langchain",
        help="Extractor to use",
    )
    args = parser.parse_args()

    # Set logging verbosity
    logging.basicConfig(
        level=(
            logging.DEBUG
            if args.verbose > 1
            else logging.INFO
            if args.verbose > 0
            else logging.WARNING
        ),
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
            LangChainExtractor(args.extractor),
        )
    )


async def main(pages: list[Path], output_path: Path, extractor: BaseExtractor) -> None:
    """Run the main program entry-point."""
    output_path.mkdir(parents=True, exist_ok=True)

    with console.status("[bold green]Extracting articles...") as status:
        async for counter, (path, article) in a.enumerate(extractor.extract_all(pages)):
            output_file = unique_path(output_path / f"{path.stem}_{article.slug}.html")
            article.write_out(output_file)
            status.update(f"Extracted {counter + 1} articles")
