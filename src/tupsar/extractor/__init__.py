"""An extractor takes an image and returns a list of articles."""

import abc
from collections.abc import AsyncIterator, Iterator

from PIL.Image import Image

from tupsar.model.article import Article


class BaseExtractor(abc.ABC):
    """Base class for all text extractors."""

    @abc.abstractmethod
    async def extract_all(self, pages: Iterator[Image]) -> AsyncIterator[Article]:
        """Extract text from a file."""
