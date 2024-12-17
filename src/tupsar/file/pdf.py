from collections.abc import Generator

import fitz
from PIL import Image
from PIL.ImageFile import ImageFile


def get_pdf_pages(pdf_path: str) -> Generator[ImageFile]:
    with fitz.open(pdf_path) as doc:
        for page_no in range(doc.page_count):
            images = doc.get_page_images(page_no, full=True)
            for image in images:
                xref = image[0]
                base_image = doc.extract_image(xref)
                yield Image.open(base_image["image"])
