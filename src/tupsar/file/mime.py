"""Helpers for handling mime types."""

from enum import Enum, auto
from pathlib import Path

import magic

mime = magic.Magic(mime=True)


class FileType(Enum):
    """Represents the type of the input file and provides helper methods."""

    IMAGE = auto()
    PDF = auto()

    @staticmethod
    def of(file_path: Path) -> "FileType":
        """
        Determine the MIME type of a file and return the corresponding FileType.

        Raises ValueError for unsupported file types.
        """
        mime_type = mime.from_file(str(file_path))

        if mime_type.startswith("image/"):
            return FileType.IMAGE
        if mime_type == "application/pdf":
            return FileType.PDF
        msg = f"Unsupported file type: {mime_type} for file {file_path}"
        raise ValueError(msg)
