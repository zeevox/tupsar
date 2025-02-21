"""An extractor takes an image and returns a list of articles."""

import abc
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from tupsar.model.article import Article


class BaseExtractor(abc.ABC):
    """Base class for all text extractors."""

    @abc.abstractmethod
    def extract_all(self, pages: Sequence[Path]) -> AsyncIterator[tuple[Path, Article]]:
        """Extract text from a file."""
