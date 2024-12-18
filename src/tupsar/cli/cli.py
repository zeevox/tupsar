"""The `tupsar` command-line interface."""

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from PIL.ImageFile import ImageFile
from rich.logging import RichHandler
from rich.progress import track
from rich_argparse import RichHelpFormatter

from tupsar.extractor.azure import AzureDocumentExtractor
from tupsar.extractor.gemini import GeminiExtractor
from tupsar.file.image import binarize_image, open_image
from tupsar.file.mime import FileType
from tupsar.file.pdf import process_pdf

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tupsar.model.article import Article

logger = logging.getLogger("tupsar")


def _process_files(file_paths: list[Path]) -> list[ImageFile]:
    processed_images: list[ImageFile] = []
    for path in file_paths:
        try:
            match FileType.of(path):
                case FileType.IMAGE:
                    img = open_image(path)
                    processed_images.append(img)
                case FileType.PDF:
                    pages = process_pdf(path)
                    processed_images.extend(pages)
        except ValueError:
            logger.exception("Could not process %s", path)
    return processed_images


def main() -> None:
    """Handle the tupsar command-line interface."""
    if not load_dotenv():
        raise FileNotFoundError

    extractors = {
        "azure": AzureDocumentExtractor,
        "gemini": GeminiExtractor,
    }

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
        default="gemini",
        help="Extractor to use",
    )
    parser.add_argument(
        "--binarize",
        action="store_true",
        help="Binarize images before processing",
        default=False,
    )
    parser.add_argument(
        "--skip-first",
        dest="start_from",
        type=int,
        help="Skip the first N pages",
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

    inputs: list[Path] = args.inputs
    pages: list[ImageFile] = _process_files(inputs)

    if args.start_from and args.start_from > len(pages):
        logger.error("Cannot skip more pages than there are")
        return

    output_path: Path = args.output_path
    extractor = extractors[args.extractor]()

    for page_no, page in enumerate(track(pages, description="Processing pages")):
        if args.start_from and page_no < args.start_from:
            continue

        articles: Iterator[Article] = extractor.extract(
            binarize_image(page) if args.binarize else page,
        )

        page_dir = output_path / f"{page_no + 1:03d}"
        page_dir.mkdir(exist_ok=True, parents=True)

        for idx, article in enumerate(articles):
            filename = f"{idx + 1:03d}_{article.slug}.md"
            article.write_out(page_dir / filename)
