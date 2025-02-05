"""Extractor implementation using LangChain."""

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Final

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables.utils import Input, Output
from PIL.Image import Image
from pydantic import BaseModel, Field

from tupsar.extractor import BaseExtractor
from tupsar.file.image import pillow_image_to_base64_string
from tupsar.model.article import Article


class _ResponseArticle(BaseModel):
    headline: str = Field(..., description="Article title")
    strapline: str | None = Field(None, description="AKA subhead or dek, if present")
    author_name: str | None = Field(None, description="Who wrote the article")
    text_body: str = Field(..., description="Text of the article, in Markdown")

    def to_article(self) -> Article:
        return Article(
            headline=self.headline,
            text_body=self.text_body,
            strapline=self.strapline,
            author_name=self.author_name,
        )


class _ResponseArticleList(BaseModel):
    articles: list[_ResponseArticle] = Field(..., description="List of articles")


class LangChainExtractor(BaseExtractor):
    """Extract text from images using the LangChain library."""

    logger = logging.getLogger(__name__)

    prompt: Final[str] = (
        "Extract all the text from this newspaper scan in the correct reading order. "
        "Source: Felix (Imperial College student newspaper) "
        "Country: United Kingdom "
        "Language: en-GB"
    )

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.2,
        max_bucket_size=3,
    )

    def __init__(self) -> None:
        """Initialise the extractor."""
        self.model = ChatAnthropic(
            model_name="claude-3-5-sonnet-20241022",
            temperature=0,
            max_tokens_to_sample=4096,
            timeout=None,
            max_retries=1,
            rate_limiter=self.rate_limiter,
        ).with_structured_output(_ResponseArticleList)

    def _get_input(self, image: Image) -> Input:
        image.thumbnail((1536, 1536))
        return [
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": pillow_image_to_base64_string(image)},
                    },
                    {"type": "text", "text": self.prompt},
                ]
            )
        ]

    async def extract_all(self, pages: Iterator[Image]) -> AsyncIterator[Article]:
        """Extract all the articles from the provided pages."""

        async def process(page: Image) -> Output | Exception:
            try:
                task: Input = self._get_input(page)
                return await self.model.ainvoke(task)
            except Exception as e:
                self.logger.exception("One of the pages failed")
                return e

        for coro in asyncio.as_completed(map(process, pages)):
            for article in (await coro).articles:
                yield article.to_article()
