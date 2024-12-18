"""Utility functions for working with PDF files."""

from collections.abc import Generator
from pathlib import Path

import fitz
from PIL import Image
from PIL.ImageFile import ImageFile


def get_pdf_pages(pdf_path: str) -> Generator[ImageFile]:
    """Load the pages of a scanned document one-by-one."""
    with fitz.open(pdf_path) as doc:
        for page_no in range(doc.page_count):
            images = doc.get_page_images(page_no, full=True)
            for image in images:
                xref = image[0]
                base_image = doc.extract_image(xref)
                yield Image.open(base_image["image"])


def process_pdf(file_path: Path) -> list[ImageFile]:
    """Load a PDF as a list of images."""
    return list(get_pdf_pages(str(file_path)))
