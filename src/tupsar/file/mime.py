from enum import Enum, auto
from pathlib import Path

import magic


class FileType(Enum):
    IMAGE = auto()
    PDF = auto()

    @staticmethod
    def of(file_path: Path) -> "FileType":
        """
        Determine the MIME type of a file and return the corresponding FileType.
        Raises ValueError for unsupported file types.
        """
        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(str(file_path))

        if mime_type.startswith("image/"):
            return FileType.IMAGE
        elif mime_type == "application/pdf":
            return FileType.PDF
        else:
            raise ValueError(f"Unsupported file type: {mime_type} for file {file_path}")
