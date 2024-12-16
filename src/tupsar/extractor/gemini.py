"""Extracts text from a scan using Google's Gemini models."""

import base64
import json
import logging
import math
import mimetypes
import os
from collections.abc import Generator
from decimal import Decimal
from pathlib import Path

from google import generativeai as genai

from tupsar.extractor.base import BaseExtractor
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

    def extract(self, path: Path) -> list[Article]:
        """Extract text from the provided scanned page using Gemini.

        Args:
            path: Path to the image file to extract text from.

        Returns:
            A list of extracted articles from the scanned page.

        """
        self.logger.info("Extracting text from %s", path)

        image_data = {
            "data": base64.b64encode(path.read_bytes()).decode("utf-8"),
            "mime_type": mimetypes.guess_file_type(path)[0],
        }

        response = self.model.generate_content(
            contents=[PROMPT, image_data],
            generation_config=(
                genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                )
            ),
        )

        if not response.text:
            # Get termination reason
            self.logger.error("No text returned from Gemini")
            self.logger.error(
                "Termination message: %s",
                response.candidates[0].finish_message,
            )
            self.logger.error(
                "Termination reason: %s",
                response.candidates[0].finish_reason,
            )
            return []

        self.logger.info("Text extracted successfully")

        log_prob: float = response.candidates[0].avg_logprobs
        lin_prob: float = round(math.exp(log_prob), 2)
        self.logger.info("Confidence: %.2f%%", lin_prob * 100)

        total_price: Decimal = self._get_response_cost(response)
        self.logger.info("Total price: $%s", total_price)

        return list(self._get_articles(response.text))

    def _get_articles(self, text: str) -> Generator[Article]:
        """Attempt to load the articles from the LLM response.

        Log and skip any malformed articles that fail to parse.

        Yields:
            Each valid article from the LLM response.

        """
        for article in json.loads(text):
            try:
                yield Article.from_json(article)
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
