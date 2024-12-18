"""Utility functions for processing images."""

import io
from pathlib import Path

import PIL.Image
from PIL.Image import Image
from PIL.ImageFile import ImageFile


def get_image_bytes(image: Image) -> bytes:
    """Return the image bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=image.format)
    return buffer.getvalue()


def open_image(file_path: Path) -> ImageFile:
    """Safely open an image file."""
    try:
        # Verify image integrity
        with PIL.Image.open(file_path) as img:
            img.verify()

        # Reopen for processing
        return PIL.Image.open(file_path)
    except (OSError, SyntaxError) as e:
        msg = "Invalid image file %s"
        raise ValueError(msg, file_path) from e
