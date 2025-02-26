"""Extractor implementation using LangChain."""

import logging
from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from pathlib import Path

from babel.numbers import format_currency
from langchain_core.runnables import Runnable, RunnableConfig

from tupsar.extractor.cost import CostTracker
from tupsar.extractor.pipeline import Model
from tupsar.model.article import Article


class LangChainExtractor:
    """Extract text from images using the LangChain library."""

    logger = logging.getLogger(__name__)

    chain: Runnable[Path, Sequence[Article]]
    cost_trackers: Sequence[CostTracker]

    def __init__(self, model_name: str, *fallback_models: str) -> None:
        """Initialise the extractor."""
        models: list[Model] = [Model(name) for name in [model_name, *fallback_models]]
        costs, chains = zip(*[model.construct_chain() for model in models], strict=True)
        primary_chain, *fallback_chains = chains

        self.chain = primary_chain

        # Add fallbacks if provided
        if fallback_chains:
            self.chain = self.chain.with_fallbacks(fallback_chains)

        self.cost_trackers = costs

    async def extract_all(
        self, pages: Sequence[Path]
    ) -> AsyncIterator[tuple[Path, Article]]:
        """Extract all the articles from the provided pages."""
        async for idx, response in self.chain.abatch_as_completed(
            pages, config=RunnableConfig(max_concurrency=12), return_exceptions=True
        ):
            page: Path = pages[idx]
            if isinstance(response, Exception):
                exception: Exception = response
                # Log error message, page and full traceback
                self.logger.error(
                    "Error processing %s: %s",
                    page,
                    exception,
                    exc_info=exception,
                )
                continue

            self.logger.debug(response)

            output: Sequence[Article] = response
            for article in output:
                yield page, article

        total_cost = sum(
            (cost_tracker.total_cost for cost_tracker in self.cost_trackers),
            start=Decimal(0),
        )
        cost_str: str = format_currency(total_cost, "USD")
        self.logger.info("Total cost: %s", cost_str)
