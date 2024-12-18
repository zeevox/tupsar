"""Extracts text from a scan using Google's Gemini models."""

import json
import logging
import math
import os
from collections.abc import Iterator
from decimal import Decimal

import google.generativeai as genai
import rich
from google.ai.generativelanguage_v1beta.types.generative_service import Candidate
from PIL.Image import Image

from tupsar.extractor import BaseExtractor
from tupsar.model.article import Article

PROMPT = """
Extract all text pieces from this scan of Felix, the Imperial College student newspaper.
Ignore advertisements.
Write in Markdown, preserving text formatting.
Fix typographical errors and remove hyphenation.
Separate paragraphs with one blank line.
""".strip()


RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "headline": {"type": "STRING"},
            "strapline": {
                "type": "STRING",
                "description": "AKA subhead or dek, if present",
            },
            "author_name": {"type": "STRING"},
            "section": {
                "type": "STRING",
                "description": "Section of the newspaper e.g. News, Opinion, Sport",
            },
            "text_body": {"type": "STRING"},
            "slug": {
                "type": "STRING",
                "description": "URL slug uniquely identifies article across issues",
            },
        },
        "required": ["headline", "section", "text_body", "slug"],
    },
}


class GeminiExtractor(BaseExtractor):
    """Extracts text from a scan using Google's Gemini models."""

    logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        """Initialise a new extractor instance."""
        genai.configure(api_key=(os.environ["GOOGLE_GEMINI_API_KEY"]))
        self.model = genai.GenerativeModel("gemini-1.5-flash-002")

    def extract(self, image: Image) -> list[Article]:
        """Extract text from the provided scanned page using Gemini.

        Args:
            image: The scanned page to extract text from.

        Returns:
            A list of extracted articles from the scanned page.

        """
        response = self.model.generate_content(
            contents=[PROMPT, image],
            generation_config=(
                genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                )
            ),
        )

        if not response.candidates:
            self.logger.error("Gemini returned an empty response.")
            return []

        # Since we only requested one candidate, indexing is safe
        candidate = response.candidates[0]

        lin_prob: float = round(math.exp(candidate.avg_logprobs), 2)
        self.logger.info("Confidence: %.2f%%", lin_prob * 100)

        total_price: Decimal = self._get_response_cost(response)
        self.logger.info("Total price: $%s", total_price)

        if (reason := candidate.finish_reason) != Candidate.FinishReason.STOP:
            self.logger.error(
                "Abnormal termination of output generation. Reason: %s (code %d)",
                reason.name,
                reason.value,
            )

            if reason == Candidate.FinishReason.RECITATION:
                sources = {
                    source.uri
                    for source in candidate.citation_metadata.citation_sources
                    if source.uri
                }
                rich.print(sources)

            return []

        self.logger.info("Text extracted successfully")

        return list(self._get_articles(response.text))

    def _get_articles(self, text: str) -> Iterator[Article]:
        """Attempt to load the articles from the LLM response.

        Log and skip any malformed articles that fail to parse.

        Yields:
            Each valid article from the LLM response.

        """
        for article in json.loads(text):
            try:
                yield Article(**article)
            except TypeError as e:
                msg = f"Error parsing article {article}: {e}"
                self.logger.exception(msg)

    @staticmethod
    def _get_response_cost(
        response: genai.types.GenerateContentResponse,
        input_rate: Decimal = Decimal("0.000000075"),  # $0.075 per million
        output_rate: Decimal = Decimal("0.0000003"),  # $0.30 per million
        cached_rate: Decimal = Decimal("0.00000001875"),  # $0.01875 per million
    ) -> Decimal:
        """Calculate the cost of a single model response.

        Args:
            response: The model response to calculate for
            input_rate: Input cost per token, in USD
            output_rate: Output cost per token, in USD
            cached_rate: Cost per cached token, in USD

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
            (normal_input_tokens * input_rate)
            + (candidates_tokens * output_rate)
            + (cached_tokens * cached_rate)
        )
