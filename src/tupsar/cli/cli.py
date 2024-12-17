import argparse
import logging
from pathlib import Path

import PIL.Image
from dotenv import load_dotenv
from PIL.ImageFile import ImageFile
from rich.logging import RichHandler
from rich.progress import track
from rich_argparse import RichHelpFormatter

from tupsar.extractor import GeminiExtractor
from tupsar.file.mime import FileType
from tupsar.file.pdf import get_pdf_pages
from tupsar.model.article import Article


def process_image(file_path: Path) -> ImageFile:
    try:
        with PIL.Image.open(file_path) as img:
            img.verify()  # Verify integrity
        return PIL.Image.open(file_path)  # Reopen for processing
    except (OSError, SyntaxError) as e:
        raise ValueError(f"Invalid image file {file_path}: {e}")


def process_pdf(file_path: Path) -> list[ImageFile]:
    return list(get_pdf_pages(str(file_path)))


def process_files(file_paths: list[Path]) -> list[ImageFile]:
    processed_images: list[ImageFile] = []
    for path in file_paths:
        try:
            match FileType.of(path):
                case FileType.IMAGE:
                    img = process_image(path)
                    processed_images.append(img)
                case FileType.PDF:
                    pages = process_pdf(path)
                    processed_images.extend(pages)
        except ValueError as ve:
            print(f"Error processing {path}: {ve}")
    return processed_images


def main() -> None:
    if not load_dotenv():
        raise FileNotFoundError

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
    pages: list[ImageFile] = process_files(inputs)

    output_path: Path = args.output_path
    output_path.mkdir(exist_ok=True, parents=True)

    articles: list[Article] = []
    for page in track(pages):
        articles.extend(GeminiExtractor().extract(page))

    # Print article text
    for idx, article in enumerate(articles):
        article_path = output_path / f"{idx}_{article.slug}.md"
        article.write_out(article_path)
