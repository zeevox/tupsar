"""Utility functions for processing images."""

import base64
import io
from pathlib import Path

import numpy as np
import PIL.Image
from PIL.Image import Image
from PIL.ImageFile import ImageFile
from skimage import filters


def get_image_bytes(image: Image) -> bytes:
    """Return the image bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=image.format)
    return buffer.getvalue()


def pillow_image_to_base64_url(img: Image) -> str:
    """
    Convert a PIL Image to a base64-encoded JPEG data URL.

    From https://stackoverflow.com/a/68989496/8459583
    """
    return f"data:image/jpeg;base64,{pillow_image_to_base64_string(img)}"


def pillow_image_to_base64_string(img: Image) -> str:
    """Encode a PIL image in base64."""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=75)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


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


def binarize_image(image: Image) -> Image:
    """
    Turn an image black-and-white.

    Args:
        image (Image): Input image.

    """
    image_np = np.array(image.copy().convert("L"))

    threshold = filters.threshold_sauvola(image_np, window_size=25, k=0.2)
    binary_image = ((image_np > threshold) * 255).astype(np.uint8)

    return PIL.Image.fromarray(binary_image)


def parse_filename(file: Path) -> tuple[str, int]:
    """Parse a Felix issue filename."""
    # parse e.g. felix_1001-001 or felix-daily-2011_2-10
    issue, page = file.stem.rsplit("-", 1)
    return issue, int(page)
