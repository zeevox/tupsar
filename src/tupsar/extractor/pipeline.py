"""Support large language model extraction pipelines."""

import enum
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL.Image import Resampling

from tupsar.extractor.parser import ArticleOutputParser
from tupsar.file.image import open_image, pillow_image_to_base64_string
from tupsar.model.article import Article


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


def prepare_image(path: Path) -> dict[str, str]:
    """Prepare an image for processing."""
    image = open_image(path)
    max_size: int = 3072
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), resample=Resampling.LANCZOS)
    return {"image_data": pillow_image_to_base64_string(image)}
