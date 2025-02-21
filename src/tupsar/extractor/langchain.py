"""Extractor implementation using LangChain."""

import enum
import logging
import re
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, override

import bs4
from bs4 import Tag
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import BaseGenerationOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from tupsar.file.image import open_image, pillow_image_to_base64_string
from tupsar.model.article import Article

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

SYSTEM_PROMPT = (
    "You are an expert typist transcribing articles from "
    "archival scans of Felix, the Imperial College London student newspaper. "
    "Structure your response as a set of one or more HTML articles. "
    "Use HTML heading and formatting tags to structure complex articles. "
    "Only start a new article when the context changes. "
    "For each <article> you identify, add the following to the <header>\n"
    "  a. <headline>: article headline\n"
    "  b. <strapline>: the subhead or dek, if specified.\n"
    "  c. <author_name>\n"
    "  d. <category>: the section of the newspaper to which the article belongs.\n"
    "Then in the <main> write the article contents in HTML format. "
    "Please reflow each article into coherent paragraphs. "
    "Ensure the transcription is as accurate as possible. "
    "Preserve original punctuation, capitalisation, and formatting. "
    "Ignore images and advertisements."
)
USER_PROMPT = "Process the entire newspaper scan. Begin the task now."

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


class LangChainExtractor:
    """Extract text from images using the LangChain library."""

    class Model(enum.StrEnum):
        """Supported large language models."""

        GEMINI_1_5 = "gemini-1.5"
        GEMINI_1_5_PRO = "gemini-1.5-pro"
        GEMINI_2_0 = "gemini-2.0"
        CLAUDE_3_5 = "claude-3.5"

        def construct_model(self) -> BaseChatModel:
            """Return an instance of the corresponding LangChain model."""
            match self:
                case self.CLAUDE_3_5:
                    return ChatAnthropic(
                        model_name="claude-3-5-sonnet-20241022",
                        temperature=0,
                        max_tokens_to_sample=8192,
                        timeout=None,
                        max_retries=0,
                        rate_limiter=InMemoryRateLimiter(
                            requests_per_second=0.2,
                            max_bucket_size=3,
                        ),
                        stop=["\n\nHuman:"],
                    )

                case self.GEMINI_1_5:
                    return ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash-002",
                        temperature=0,
                        max_tokens=8192,
                        timeout=None,
                        max_retries=1,
                        rate_limiter=InMemoryRateLimiter(
                            requests_per_second=30,
                            max_bucket_size=30,
                        ),
                    )

                case self.GEMINI_1_5_PRO:
                    return ChatGoogleGenerativeAI(
                        model="gemini-1.5-pro-002",
                        temperature=0,
                        max_tokens=8192,
                        timeout=None,
                        max_retries=1,
                        rate_limiter=InMemoryRateLimiter(
                            requests_per_second=15,
                            max_bucket_size=15,
                        ),
                    )

                case self.GEMINI_2_0:
                    return ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash-001",
                        temperature=0,
                        max_tokens=8192,
                        timeout=None,
                        max_retries=1,
                        rate_limiter=InMemoryRateLimiter(
                            requests_per_second=30,
                            max_bucket_size=30,
                        ),
                    )

            raise ValueError

        def construct_chain(self) -> Runnable[Path, Sequence[Article]]:
            """Return an instance of the full corresponding extraction pipeline."""
            return (
                RunnableLambda(prepare_image)
                | PROMPT_TEMPLATE
                | self.construct_model()
                | ArticleOutputParser()
            )

        @property
        def max_image_input_size(self) -> tuple[int, int]:
            """Get the maximum input image size the model supports."""
            match self:
                case self.CLAUDE_3_5:
                    return 1536, 1536
                case self.GEMINI_1_5 | self.GEMINI_1_5_PRO | self.GEMINI_2_0:
                    return 3072, 3072
            raise ValueError

    logger = logging.getLogger(__name__)

    def __init__(self, model: str | Model) -> None:
        """Initialise the extractor."""
        if isinstance(model, str):
            model = self.Model(model)

        self.model = model.construct_chain().with_fallbacks([
            self.Model.GEMINI_1_5_PRO.construct_chain()
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


class ArticleOutputParser(BaseGenerationOutputParser[Sequence[Article]]):
    """Custom LangChain parser yields articles."""

    @override
    def parse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> Sequence[Article]:
        """Parse the response into an article."""
        if len(result) != 1:
            msg = "Expected exactly one response"
            raise NotImplementedError(msg)
        generation: Generation = result[0]

        if not isinstance(generation, ChatGeneration):
            msg = "This output parser can only be used with a chat generation."
            raise OutputParserException(msg)

        response: BaseMessage = generation.message
        response_metadata: dict[str, Any] = response.response_metadata

        finish_reason = (
            response_metadata.get("finish_reason")
            or response_metadata.get("stop_reason")
            or "missing"
        ).lower()
        if finish_reason not in {"stop", "end_turn"}:
            msg = f"Unexpected finish reason {finish_reason}"
            raise OutputParserException(msg)

        feedback = response_metadata.get("prompt_feedback", {})
        if (block_reason := feedback.get("block_reason", 0)) != 0:
            msg = f"Extraction blocked with code {block_reason}"
            raise OutputParserException(msg)

        if not isinstance(response.content, str):
            msg = "Expected response content to be a string"
            raise OutputParserException(msg)

        soup = bs4.BeautifulSoup(_get_llm_xml(response.content), "lxml")
        articles = soup.find_all("article", recursive=True)
        if not articles:
            msg = "No articles returned"
            raise OutputParserException(msg)

        return [
            Article(
                headline=_get_text(article, "headline") or "Untitled",
                text_body=article.find("main"),
                strapline=_get_text(article, "strapline"),
                author_name=_get_text(article, "author_name"),
                category=_get_text(article, "category"),
            )
            for article in articles
        ]


def prepare_image(path: Path) -> dict[str, str]:
    """Prepare an image for processing."""
    image = open_image(path)
    image.thumbnail((3072, 3072))
    return {"image_data": pillow_image_to_base64_string(image)}


def _get_text(tree: Tag, query: str) -> str | None:
    element = tree.find(query)
    if element is not None:
        text: str = element.get_text(strip=True)
        return text
    return None


encoding_matcher: re.Pattern[str] = re.compile(
    r"<([^>]*encoding[^>]*)>\n(.*)", re.MULTILINE | re.DOTALL
)


def _get_llm_xml(text: str) -> str:
    # From https://github.com/langchain-ai/langchain/blob/037b129b86eaf0ba077b406bfa81fb4059d35874/libs/core/langchain_core/output_parsers/xml.py#L219-L227
    match = re.search(r"```(html|xml)?(.*)```", text, re.DOTALL)
    if match is not None:
        # If match found, use the content within the backticks
        text = match.group(2)
    encoding_match = encoding_matcher.search(text)
    if encoding_match:
        text = encoding_match.group(2)

    return f"<html>{text.strip()}</html>"
