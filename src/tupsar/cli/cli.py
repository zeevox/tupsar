"""The `tupsar` command-line interface."""

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path

from dotenv import load_dotenv
from PIL.ImageFile import ImageFile
from rich.logging import RichHandler
from rich_argparse import RichHelpFormatter

from tupsar.extractor import BaseExtractor
from tupsar.extractor.azure import AzureDocumentExtractor
from tupsar.extractor.gemini import GeminiExtractor
from tupsar.extractor.langchain import LangChainExtractor
from tupsar.file.image import open_image
from tupsar.file.mime import FileType
from tupsar.file.pdf import process_pdf

logger = logging.getLogger("tupsar")


def _process_files(file_paths: list[Path]) -> Iterator[ImageFile]:
    for path in file_paths:
        try:
            match FileType.of(path):
                case FileType.IMAGE:
                    yield open_image(path)
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

    asyncio.run(main(args.inputs, args.output_path, extractors[args.extractor]()))


async def main(inputs: list[Path], output_path: Path, extractor: BaseExtractor) -> None:
    """Run the main program entry-point."""
    pages: Iterator[ImageFile] = _process_files(inputs)
    output_path.mkdir(parents=True, exist_ok=True)

    async for article in extractor.extract_all(pages):
        article_hash = hashlib.sha256(
            json.dumps(dataclasses.asdict(article), sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]
        filename = f"{article.slug}_{article_hash}.md"
        article.write_out(output_path / filename)
