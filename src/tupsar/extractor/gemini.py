"""Extracts text from a scan using Google's Gemini models."""

import json
import logging
import math
import os
from collections.abc import Iterator
from decimal import Decimal

import google.generativeai as genai
import ijson
import rich
from google.ai.generativelanguage_v1beta import HarmCategory
from google.ai.generativelanguage_v1beta.types.generative_service import Candidate
from google.generativeai.types import HarmBlockThreshold as Threshold
from PIL.Image import Image
from rich.status import Status

from tupsar.extractor import BaseExtractor
from tupsar.file.image import binarize_image
from tupsar.model.article import Article

PROMPT = """
Extract all text pieces from this scan of Felix, the Imperial College student newspaper
Ignore advertisements
Fix any typos
Country: United Kingdom
Language: en-GB
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
            "text_body": {
                "type": "STRING",
                "description": "The text of the article, in Markdown",
            },
        },
        "required": ["headline", "text_body"],
    },
}


class GeminiExtractor(BaseExtractor):
    """Extracts text from a scan using Google's Gemini models."""

    logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        """Initialise a new extractor instance."""
        genai.configure(api_key=(os.environ["GOOGLE_GEMINI_API_KEY"]))
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")

    def extract(self, image: Image) -> Iterator[Article]:
        """Extract text from the provided scanned page using Gemini.

        Args:
            image: The scanned page to extract text from.

        Returns:
            A list of extracted articles from the scanned page.

        """
        status = Status("Extracting text from scan...")
        status.start()

        # Resize the image to 3072 pixels on the longest side
        status.update("Resizing image...")
        image.thumbnail((3072, 3072))

        # Binarize the image
        status.update("Binarizing image...")
        binarized = binarize_image(image)

        status.update("Calling Gemini...")
        response = self.model.generate_content(
            # Google says to place the prompt after the image for best results.
            contents=[binarized, PROMPT],
            stream=True,
            generation_config=(
                genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                )
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: Threshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: Threshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: Threshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: Threshold.BLOCK_NONE,
            },
        )

        events = ijson.sendable_list()
        coro = ijson.items_coro(events, "item")

        total_tokens: int = 0
        for chunk in response:
            if not chunk.candidates[0].content.parts:
                self.logger.warning("Received no chunks")
                continue

            chunk_text: str = chunk.candidates[0].content.parts[0].text
            if not chunk_text:
                self.logger.warning("Received empty chunk")
                continue

            self.logger.debug("Received chunk: %s", chunk_text)
            total_tokens += self.model.count_tokens(chunk_text).total_tokens
            status.update(f"Generated {total_tokens} tokens...")

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

        if not response.candidates:
            self.logger.error("Gemini returned an empty response.")
            raise StopIteration

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

            raise StopIteration

        status.stop()
        self.logger.info("Text extracted successfully")

    def _get_articles(self, text: str) -> Iterator[Article]:
        """Attempt to load the articles from the LLM response.

        Log and skip any malformed articles that fail to parse.

        Yields:
            Each valid article from the LLM response.

        """
        for article in json.loads(text):
            try:
                yield Article(
                    headline=article["headline"],
                    text_body=article["text_body"],
                    strapline=article.get("strapline"),
                    author_name=article.get("author_name"),
                )
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
