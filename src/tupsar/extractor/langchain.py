"""Extractor implementation using LangChain."""

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from enum import StrEnum
from typing import Final

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable
from langchain_core.runnables.utils import Output
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL.Image import Image
from pydantic import BaseModel, Field

from tupsar.extractor import BaseExtractor
from tupsar.file.image import pillow_image_to_base64_string
from tupsar.model.article import Article

SYSTEM_PROMPT = (
    "You are an expert typist tasked with transcribing the articles from "
    "archived copies of Felix, the Imperial College London student newspaper. "
    "Structure your response in JSON as an array of newspaper articles. "
    "Please follow these steps to extract and structure the text:\n"
    " 1. Identify individual articles within the newspaper scan.\n"
    " 2. For each newspaper article extract the following fields:\n"
    "  a. headline: article headline, as a string\n"
    "  b. strapline: AKA subhead or dek (if present)\n"
    "  c. author name: who wrote the article (if specified)\n"
    "  d. text_body: the text of the article\n"
    "Every article must have a headline and text_body field.\n"
    "Use Markdown formatting for the main text of each article. "
    "For example, _italicise_ or **bold** as it appears in the scan, "
    "start article sub-headings with `##`, and bullet points with `*`. "
    "For each article, please reflow the text_body into coherent paragraphs. "
    "Remove all extraneous line breaks and "
    "merge any hyphenated words split across lines "
    "so that the text reads as normal prose."
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

    class Model(StrEnum):
        """Supported large language models."""

        GEMINI = "gemini-1.5-flash-002"
        CLAUDE = "claude-3-5-sonnet-20241022"

        def construct_model(self) -> BaseChatModel:
            """Return an instance of the corresponding LangChain model."""
            match self:
                case self.CLAUDE:
                    return ChatAnthropic(
                        model_name=self,
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
                        model="gemini-1.5-flash-002",
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

    async def extract_all(self, pages: Iterator[Image]) -> AsyncIterator[Article]:
        """Extract all the articles from the provided pages."""

        async def process(page: Image) -> Output | Exception:
            try:
                page.thumbnail(self.Model.GEMINI.max_image_input_size)
                return await self.model.ainvoke({
                    "image_data": pillow_image_to_base64_string(page)
                })
            except Exception as e:
                self.logger.exception("One of the pages failed")
                return e

        for coro in asyncio.as_completed(map(process, pages)):
            response: dict = await coro
            self.logger.debug(response)

            for article in response:
                if not isinstance(article, dict):
                    self.logger.error("Got unexpected article response: %s", article)
                    continue

                yield Article(
                    headline=article.get("headline") or "Untitled",
                    text_body=article.get("text_body") or "[Empty article]",
                    strapline=article.get("strapline"),
                    author_name=article.get("author_name"),
                )
