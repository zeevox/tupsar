"""A single page of an issue."""

import dataclasses
from pathlib import Path

from PIL import ImageFile


def parse_filename(file: Path) -> tuple[str, int]:
    """Parse a Felix issue filename."""
    # parse e.g. felix_1001-001 or felix-daily-2011_2-10
    issue, page = file.stem.rsplit("-", 1)
    return issue, int(page)


@dataclasses.dataclass
class Page:
    """A single page of an issue."""

    issue: str
    page_no: int
    image: ImageFile
