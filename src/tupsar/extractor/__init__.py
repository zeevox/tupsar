"""An extractor takes an image and returns a list of articles."""

import abc

from PIL.Image import Image

from tupsar.model.article import Article


class BaseExtractor(abc.ABC):
    """Base class for all text extractors."""

    @abc.abstractmethod
    def extract(self, image: Image) -> list[Article]:
        """Extract text from a file."""
