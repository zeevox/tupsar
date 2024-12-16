"""Base class for all text extractors."""

import abc
from pathlib import Path

from tupsar.model.article import Article


class BaseExtractor(abc.ABC):
    """Base class for all text extractors."""

    @abc.abstractmethod
    def extract(self, path: Path) -> list[Article]:
        """Extract text from a file."""
