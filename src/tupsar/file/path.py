"""Utility functions for working with file paths."""

from itertools import count
from pathlib import Path


def unique_path(path: Path) -> Path:
    """Add a counter to the filename to make it unique."""
    return next(
        candidate
        for i in count()
        if not (
            candidate := path.with_stem(f"{path.stem}{'' if i == 0 else f'_{i}'}")
        ).exists()
    )
