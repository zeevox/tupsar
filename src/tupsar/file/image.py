"""Utility functions for processing images."""

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
