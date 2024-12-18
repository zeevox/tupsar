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
from tupsar.file.image import open_image
from tupsar.file.mime import FileType
from tupsar.file.pdf import process_pdf

if TYPE_CHECKING:
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

    output_path: Path = args.output_path
    output_path.mkdir(exist_ok=True, parents=True)

    extractor = extractors[args.extractor]()

    articles: list[Article] = []
    for page in track(pages):
        articles.extend(extractor.extract(page))

    # Print article text
    for idx, article in enumerate(articles):
        article_path = output_path / f"{idx}_{article.slug}.md"
        article.write_out(article_path)
