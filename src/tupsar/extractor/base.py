"""Base class for all text extractors."""

import abc
from pathlib import Path


class BaseExtractor(abc.ABC):
    """Base class for all text extractors."""

    @abc.abstractmethod
    def extract(self, path: Path) -> str:
        """Extract text from a file."""
