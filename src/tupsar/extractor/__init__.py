"""An extractor takes an image and returns a list of articles."""

import abc
from collections.abc import AsyncIterator, Iterator

from tupsar.model.article import Article
from tupsar.model.page import Page


class BaseExtractor(abc.ABC):
    """Base class for all text extractors."""

    @abc.abstractmethod
    async def extract_all(self, pages: Iterator[Page]) -> AsyncIterator[Article]:
        """Extract text from a file."""
