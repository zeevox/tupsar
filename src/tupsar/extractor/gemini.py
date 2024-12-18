"""Extracts text from a scan using Google's Gemini models."""

import dataclasses
import logging
import math
import os
from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import google.generativeai as genai
import ijson
from google.ai.generativelanguage_v1beta import HarmCategory
from google.ai.generativelanguage_v1beta.types.generative_service import Candidate
from google.api_core.exceptions import InternalServerError
from google.generativeai.types import HarmBlockThreshold as Threshold
from PIL.Image import Image

from tupsar.extractor import BaseExtractor
from tupsar.model.article import Article

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "description": "A newspaper article",
        "properties": {
            "headline": {"type": "STRING"},
            "strapline": {
                "type": "STRING",
                "description": "AKA subhead or dek, if present",
            },
            "author_name": {"type": "STRING"},
            "text_body": {
                "type": "STRING",
                "description": "The text of the article, in Markdown",
            },
        },
        "required": ["headline", "text_body"],
    },
}


class GeminiGenerationError(BaseException):
    """Base exception for Gemini-related errors."""


@dataclasses.dataclass
class _GeminiModel:
    name: str

    input_rate: Decimal = Decimal("0.000000075")  # $0.075 per million
    """Input cost per token, in USD"""

    output_rate: Decimal = Decimal("0.0000003")  # $0.30 per million
    """Output cost per token, in USD"""

    cached_rate: Decimal = Decimal("0.00000001875")  # $0.01875 per million
    """Cost per cached token, in USD"""

    prompt: str = (
        "Extract all the text from this newspaper scan."
        "Ignore advertisements."
        "Fix any typos."
        "Source: Felix (Imperial College student newspaper)"
        "Country: United Kingdom"
        "Language: en-GB"
    )

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.model = genai.GenerativeModel(
            self.name,
            generation_config=(
                genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                )
            ),
            # Since this is a text extraction job, disable all content filters
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: Threshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: Threshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: Threshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: Threshold.BLOCK_NONE,
            },
        )

    def generate_response(self, image: Image) -> Iterator[str]:
        response = self.model.generate_content(
            # Google says to place the prompt after the image for best results.
            contents=[image, self.prompt],
            # Stream the response to yield articles as soon as they are ready
            stream=True,
        )

        for chunk in response:
            if not chunk.candidates:
                self.logger.warning("Received no candidates")
                continue

            candidate = chunk.candidates[0]

            if not candidate.content.parts:
                self.logger.warning("Received no content parts")
                continue

            chunk_text: str = chunk.candidates[0].content.parts[0].text

            if not chunk_text:
                self.logger.warning("Received empty chunk")
                continue

            self.logger.debug("Received chunk: %s", chunk_text)

            # Yield the chunk to the caller
            yield chunk_text

        if not response.candidates:
            raise GeminiGenerationError

        # Since we only requested one candidate, indexing is safe
        candidate = response.candidates[0]

        total_price: Decimal = self._get_response_cost(response)
        self.logger.info("Total price: $%s", total_price)

        if (reason := candidate.finish_reason) != Candidate.FinishReason.STOP:
            extras: dict[str, Any] = {}
            if reason == Candidate.FinishReason.RECITATION:
                extras["sources"] = {
                    source.uri
                    for source in candidate.citation_metadata.citation_sources
                    if source.uri
                }

            self.logger.error(
                "Abnormal termination of output generation. Reason: %s (code %d)",
                reason.name,
                reason.value,
                extra=extras or None,
            )

            raise GeminiGenerationError

        lin_prob: float = round(math.exp(candidate.avg_logprobs), 2)
        self.logger.info("Confidence: %.2f%%", lin_prob * 100)

    def _get_response_cost(
        self,
        response: genai.types.GenerateContentResponse,
    ) -> Decimal:
        """Calculate the cost of a single model response.

        Args:
            response: The model response to calculate for

        Returns:
            The cost of generating the response in USD.

        """
        prompt_tokens = Decimal(response.usage_metadata.prompt_token_count or 0)
        candidates_tokens = Decimal(response.usage_metadata.candidates_token_count or 0)
        cached_tokens = Decimal(response.usage_metadata.cached_content_token_count or 0)

        normal_input_tokens = prompt_tokens - cached_tokens
        if normal_input_tokens < 0:
            normal_input_tokens = Decimal(0)

        return (
            (normal_input_tokens * self.input_rate)
            + (candidates_tokens * self.output_rate)
            + (cached_tokens * self.cached_rate)
        )


class GeminiExtractor(BaseExtractor):
    """Extracts text from a scan using Google's Gemini models."""

    logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        """Initialise a new extractor instance."""
        genai.configure(api_key=(os.environ["GOOGLE_GEMINI_API_KEY"]))

    def _extract(self, model: _GeminiModel, image: Image) -> Iterator[Article]:
        # Resize the image to 3072 pixels on the longest side
        image.thumbnail((3072, 3072))

        events = ijson.sendable_list()
        coro = ijson.items_coro(events, "item")

        for chunk_text in model.generate_response(image):
            coro.send(chunk_text.encode("utf-8"))

            # Yield any fully parsed articles
            for parsed_obj in events:
                yield Article(
                    headline=parsed_obj["headline"],
                    text_body=parsed_obj["text_body"],
                    strapline=parsed_obj.get("strapline"),
                    author_name=parsed_obj.get("author_name"),
                )

            # Clear the list to prepare for the next batch
            del events[:]

        coro.close()

        if events:
            self.logger.warning("Leftover objects remain after parsing")

        self.logger.info("Text extracted successfully")

    def extract(self, image: Image) -> Iterator[Article]:
        """Extract text from the provided scanned page using Gemini.

        Args:
            image: The scanned page to extract text from.

        Returns:
            A list of extracted articles from the scanned page.

        """
        models = [
            _GeminiModel(
                "gemini-1.5-flash-002",
                input_rate=Decimal("0.000000075"),  # $0.075 per million
                output_rate=Decimal("0.0000003"),  # $0.30 per million
                cached_rate=Decimal("0.00000001875"),  # $0.01875 per million
            ),
            _GeminiModel(
                "gemini-2.0-flash-exp",
                input_rate=Decimal(0),
                output_rate=Decimal(0),
                cached_rate=Decimal(0),
            ),
        ]

        # Use the first model that does not raise a GeminiGenerationError
        for model in models:
            try:
                yield from self._extract(model, image)
                break
            except (GeminiGenerationError, InternalServerError):
                self.logger.warning("Model %s failed to generate output", model.name)
        else:
            self.logger.error("All models failed to generate output; skipping page.")
            raise StopIteration
