"""The `tupsar` command-line interface."""

import argparse
import asyncio
import logging
from collections.abc import Iterator
from pathlib import Path

import asyncstdlib as a
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich_argparse import RichHelpFormatter

from tupsar.extractor import BaseExtractor
from tupsar.extractor.azure import AzureDocumentExtractor
from tupsar.extractor.gemini import GeminiExtractor
from tupsar.extractor.langchain import LangChainExtractor
from tupsar.file.image import open_image
from tupsar.file.mime import FileType
from tupsar.file.path import unique_path
from tupsar.file.pdf import process_pdf
from tupsar.model.page import Page, parse_filename

logger = logging.getLogger("tupsar")


def _process_files(file_paths: list[Path]) -> Iterator[Page]:
    for path in file_paths:
        try:
            match FileType.of(path):
                case FileType.IMAGE:
                    issue, page_no = parse_filename(path)
                    yield Page(issue, page_no, open_image(path))
                case FileType.PDF:
                    yield from process_pdf(path)
        except ValueError:
            logger.exception("Could not process %s", path)


extractors = {
    "azure": AzureDocumentExtractor,
    "gemini": GeminiExtractor,
    "langchain": LangChainExtractor,
}


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
        choices=extractors.keys(),
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
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    asyncio.run(
        main(
            args.inputs,
            args.output_path,
            LangChainExtractor(
                LangChainExtractor.Model.GEMINI,
                LangChainExtractor.Model.CLAUDE,
            ),
        )
    )


async def main(inputs: list[Path], output_path: Path, extractor: BaseExtractor) -> None:
    """Run the main program entry-point."""
    pages: Iterator[Page] = _process_files(inputs)
    output_path.mkdir(parents=True, exist_ok=True)

    console = Console()

    with console.status("[bold green]Extracting articles...") as status:
        async for counter, article in a.enumerate(extractor.extract_all(pages)):
            output_file = unique_path(
                output_path
                / f"{article.issue}-{article.page_no:03d}_{article.slug}.html"
            )
            article.write_out(output_file)
            status.update(f"Extracted {counter + 1} articles")
