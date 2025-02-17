"""Utility functions for working with PDF files."""

import io
from collections.abc import Iterator
from pathlib import Path

import fitz
from PIL import Image
from PIL.ImageFile import ImageFile

from tupsar.model.page import Page


def get_pdf_pages(pdf_path: str | Path) -> Iterator[ImageFile]:
    """Load the pages of a scanned document one-by-one."""
    with fitz.open(pdf_path) as doc:
        for page_no in range(doc.page_count):
            images = doc.get_page_images(page_no, full=True)
            for image in images:
                xref = image[0]
                base_image = doc.extract_image(xref)
                yield Image.open(io.BytesIO(base_image["image"]))


def process_pdf(file_path: Path) -> Iterator[Page]:
    """Load a PDF as a list of images."""
    for page_no, image in enumerate(get_pdf_pages(file_path)):
        yield Page(file_path.stem, page_no, image)
