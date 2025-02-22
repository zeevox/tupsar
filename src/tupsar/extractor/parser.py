"""Output parser turns LLM responses into articles."""

import logging
import re
from collections.abc import Sequence
from logging import Logger
from typing import TYPE_CHECKING, Any, override

import bs4
from bs4 import Tag
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseGenerationOutputParser
from langchain_core.outputs import ChatGeneration, Generation

from tupsar.model.article import Article

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


class ArticleOutputParser(BaseGenerationOutputParser[Sequence[Article]]):
    """Custom LangChain parser yields articles."""

    @property
    def logger(self) -> Logger:
        """Return the logger."""
        return logging.getLogger(__name__)

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
        self.logger.debug(response)

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
