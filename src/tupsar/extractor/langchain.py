"""Extractor implementation using LangChain."""

import asyncio
import enum
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Final

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable
from langchain_core.runnables.utils import Output
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from tupsar.extractor import BaseExtractor
from tupsar.file.image import pillow_image_to_base64_string
from tupsar.model.article import Article
from tupsar.model.page import Page

SYSTEM_PROMPT = (
    "You are an expert typist transcribing articles from "
    "archival scans of Felix, the Imperial College London student newspaper. "
    "Structure your response as a JSON array of articles. "
    "Use Markdown headings and formatting to structure complex articles, "
    "rather than splitting them up. "
    "For each article you identify, create a JSON object with these fields:\n"
    "  a. headline: article headline (required)\n"
    "  b. strapline: the subhead or dek, if specified.\n"
    "  c. author_name\n"
    "  d. text_body: the article contents, in Markdown format (required)\n"
    "  e. category: the section of the newspaper to which the article belongs.\n"
    "For each article, please reflow the text_body into coherent paragraphs. "
    "Ensure the transcription is as accurate as possible. "
    "Preserve original punctuation, capitalisation, and formatting. "
)
USER_PROMPT = (
    "Process the entire newspaper scan and structure all articles in this format. "
    "Begin the task now."
)

PROMPT_TEMPLATE: Final[ChatPromptTemplate] = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    (
        "user",
        [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
            },
            {"type": "text", "text": USER_PROMPT},
        ],
    ),
])


class _ResponseArticle(BaseModel):
    headline: str = Field(..., description="Article title")
    strapline: str | None = Field(None, description="AKA subhead or dek, if present")
    author_name: str | None = Field(None, description="Who wrote the article")
    text_body: str = Field(..., description="Text of the article, in Markdown")
    category: str | None = Field(
        None, description="The section of the newspaper to which the article belongs"
    )

    def to_article(self) -> Article:
        return Article(
            headline=self.headline,
            text_body=self.text_body,
            strapline=self.strapline,
            author_name=self.author_name,
            category=self.category,
        )


class _ResponseArticleList(BaseModel):
    articles: list[_ResponseArticle] = Field(..., description="List of articles")


class LangChainExtractor(BaseExtractor):
    """Extract text from images using the LangChain library."""

    class Model(enum.StrEnum):
        """Supported large language models."""

        GEMINI = enum.auto()
        CLAUDE = enum.auto()

        def construct_model(self) -> BaseChatModel:
            """Return an instance of the corresponding LangChain model."""
            match self:
                case self.CLAUDE:
                    return ChatAnthropic(
                        model_name="claude-3-5-sonnet-20241022",
                        temperature=0,
                        max_tokens_to_sample=4096,
                        timeout=None,
                        max_retries=0,
                        rate_limiter=InMemoryRateLimiter(
                            requests_per_second=0.2,
                            max_bucket_size=3,
                        ),
                    )

                case self.GEMINI:
                    return ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash-001",
                        temperature=0,
                        max_tokens=4096,
                        timeout=None,
                        max_retries=1,
                        rate_limiter=InMemoryRateLimiter(
                            requests_per_second=20,
                            max_bucket_size=15,
                        ),
                    )

            raise ValueError

        def construct_chain(self) -> Runnable:
            """Return an instance of the full corresponding extraction pipeline."""
            return (
                PROMPT_TEMPLATE
                | self.construct_model()
                | JsonOutputParser(pydantic_object=_ResponseArticleList)
            )

        @property
        def max_image_input_size(self) -> tuple[int, int]:
            """Get the maximum input image size the model supports."""
            match self:
                case self.CLAUDE:
                    return 1536, 1536
                case self.GEMINI:
                    return 3072, 3072
            raise ValueError

    logger = logging.getLogger(__name__)

    def __init__(self, *models: Model) -> None:
        """Initialise the extractor."""
        primary, *fallbacks = models

        self.model = (
            primary.construct_model()
            if not fallbacks
            else primary.construct_chain().with_fallbacks([
                fallback.construct_chain() for fallback in fallbacks
            ])
        )

    async def extract_all(self, pages: Iterator[Page]) -> AsyncIterator[Article]:
        """Extract all the articles from the provided pages."""

        async def process(input_page: Page) -> tuple[Page, Output | Exception]:
            try:
                input_page.image.thumbnail(self.Model.GEMINI.max_image_input_size)
                return input_page, await self.model.ainvoke({
                    "image_data": pillow_image_to_base64_string(input_page.image)
                })
            except Exception as e:
                self.logger.exception(
                    "%s page %d failed", input_page.issue, input_page.page_no
                )
                return input_page, e

        for coro in asyncio.as_completed(map(process, pages)):
            page, response = await coro
            self.logger.debug(response)

            for article in response:
                if not isinstance(article, dict):
                    self.logger.error("Got unexpected article response: %s", article)
                    continue

                yield Article(
                    issue=page.issue,
                    page_no=page.page_no,
                    headline=article.get("headline") or "Untitled",
                    text_body=article.get("text_body") or "[Empty article]",
                    strapline=article.get("strapline"),
                    author_name=article.get("author_name"),
                    category=article.get("category"),
                )
