"""Support large language model extraction pipelines."""

import enum
import os
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Final

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils import convert_to_secret_str

from tupsar.extractor.cost import CostTracker
from tupsar.extractor.parser import ArticleOutputParser
from tupsar.file.image import open_image, pillow_image_to_base64_string
from tupsar.model.article import Article


class Model(enum.StrEnum):
    """Supported large language models."""

    GEMINI_1_5 = "gemini-1.5"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_2_0 = "gemini-2.0"
    CLAUDE_3_7 = "claude-3.7"
    GPT_4_1 = "gpt-4.1"
    QWEN_2_5_VL_72B_INSTRUCT = "qwen-2.5-vl"

    def construct_model(self) -> BaseChatModel:
        """Return an instance of the corresponding LangChain model."""
        match self:
            case self.CLAUDE_3_7:
                return self.create_openrouter_model("anthropic/claude-3.7-sonnet")

            case self.GEMINI_1_5:
                from langchain_google_genai import ChatGoogleGenerativeAI

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
                from langchain_google_genai import ChatGoogleGenerativeAI

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
                from langchain_google_genai import ChatGoogleGenerativeAI

                return ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-001",
                    temperature=0.2,
                    max_tokens=8192,
                    timeout=None,
                    max_retries=1,
                    rate_limiter=InMemoryRateLimiter(
                        requests_per_second=30,
                        max_bucket_size=30,
                    ),
                )

            case self.GPT_4_1:
                return self.create_openrouter_model("openai/gpt-4.1", temperature=1.0)

            case self.QWEN_2_5_VL_72B_INSTRUCT:
                return self.create_openrouter_model("qwen/qwen2.5-vl-72b-instruct")
        raise ValueError

    @staticmethod
    def create_openrouter_model(
        model_name: str, **overrides: float | str
    ) -> BaseChatModel:
        """Create a model provided through OpenRouter."""
        from langchain_openai import ChatOpenAI

        api_key: str | None = os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            msg = "OPENROUTER_API_KEY environment variable not set"
            raise ValueError(msg)
        params = dict(  # noqa: C408
            api_key=convert_to_secret_str(api_key),
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
            temperature=0.2,
            top_p=0.8,
            max_completion_tokens=16384,
        )
        return ChatOpenAI(**{**params, **overrides})

    def get_cost_tracker(self) -> CostTracker:
        """Return the cost tracker for the model."""
        match self:
            case self.GEMINI_2_0:
                # Input $0.10 Output $0.40 / MTok
                return CostTracker(
                    Decimal("0.10") / 1_000_000,
                    Decimal("0.40") / 1_000_000,
                )
            case self.GEMINI_1_5_PRO:
                # Input $1.25 Output $5.00 / MTok
                return CostTracker(
                    Decimal("1.25") / 1_000_000,
                    Decimal("5.00") / 1_000_000,
                )
            case self.GEMINI_1_5:
                # Input $0.075 Output $0.30 / MTok
                return CostTracker(
                    Decimal("0.075") / 1_000_000,
                    Decimal("0.30") / 1_000_000,
                )
            case self.CLAUDE_3_7:
                # Input $3 Output $15 / MTok
                return CostTracker(
                    Decimal(3) / 1_000_000,
                    Decimal(15) / 1_000_000,
                )
            case self.QWEN_2_5_VL_72B_INSTRUCT:
                # Input $0.7 Output $0.7 / MTok
                return CostTracker(
                    Decimal("0.7") / 1_000_000,
                    Decimal("0.7") / 1_000_000,
                )
            case self.GPT_4_1:
                # Input: $3.00 / 1M tokens Output: $12.00 / 1M tokens
                return CostTracker(
                    Decimal("3.00") / 1_000_000,
                    Decimal("12.00") / 1_000_000,
                )
        raise ValueError

    def construct_chain(self) -> tuple[CostTracker, Runnable[Path, Sequence[Article]]]:
        """Return an instance of the full corresponding extraction pipeline."""
        cost_tracker = self.get_cost_tracker()
        return cost_tracker, (
            RunnableLambda(prepare_image)
            | PROMPT_TEMPLATE
            | self.construct_model()
            | ArticleOutputParser(cost_tracker)
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
    "  c. <author_name>: person who wrote the article, or signatory for letters. "
    "  d. <category>: the section of the newspaper to which the article belongs.\n"
    "Then in the <main> write the article contents in HTML format. "
    "Please reflow each article into coherent paragraphs. "
    "Ensure the transcription is as accurate as possible. "
    "Preserve original punctuation, capitalisation, and formatting. "
    "Ignore images, image captions, advertisements and puzzles like crosswords. "
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
    return {"image_data": pillow_image_to_base64_string(image)}
