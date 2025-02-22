"""Extractor implementation using LangChain."""

import logging
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from tupsar.extractor.pipeline import Model
from tupsar.model.article import Article


class LangChainExtractor:
    """Extract text from images using the LangChain library."""

    logger = logging.getLogger(__name__)

    def __init__(self, model: str | Model) -> None:
        """Initialise the extractor."""
        if isinstance(model, str):
            model = Model(model)

        self.model = model.construct_chain().with_fallbacks([
            Model.GEMINI_1_5_PRO.construct_chain()
        ])

    async def extract_all(
        self, pages: Sequence[Path]
    ) -> AsyncIterator[tuple[Path, Article]]:
        """Extract all the articles from the provided pages."""
        async for idx, response in self.model.abatch_as_completed(
            pages, config=RunnableConfig(max_concurrency=12), return_exceptions=True
        ):
            page: Path = pages[idx]
            if isinstance(response, Exception):
                self.logger.error("Extraction failed for %s: %s", page, response)
                continue

            self.logger.debug(response)

            output: Sequence[Article] = response
            for article in output:
                yield page, article
