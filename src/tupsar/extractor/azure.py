"""Extract text from an image using Azure Document Intelligence."""

import dataclasses
import logging
import os

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    DocumentAnalysisFeature,
    ParagraphRole,
)
from azure.core.credentials import AzureKeyCredential
from PIL.Image import Image

from tupsar.extractor import BaseExtractor
from tupsar.file.image import get_image_bytes
from tupsar.model.article import Article


class AzureDocumentExtractor(BaseExtractor):
    """Extract text from an image using Azure Document Intelligence."""

    def __init__(self) -> None:
        """Initialise the Azure Document Intelligence client.

        Provide the endpoint and API key as environment variables.
        """
        self.logger = logging.getLogger("azure")
        self.client = DocumentIntelligenceClient(
            endpoint=os.environ["DOCUMENTINTELLIGENCE_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["DOCUMENTINTELLIGENCE_API_KEY"]),
            api_version="2024-11-30",
        )

    @dataclasses.dataclass
    class _DocumentSection:
        title: str
        paragraphs: list[str] = dataclasses.field(default_factory=list)

        def to_article(self) -> Article:
            return Article(
                headline=self.title,
                text_body="\n\n".join(self.paragraphs),
            )

    def extract(self, image: Image) -> list[Article]:
        """Extract text from an image using Azure Document Intelligence."""
        self.logger.info(
            "Extracting text from %s (%s) using Azure Document Intelligence",
            image,
            image.format,
        )
        poller = self.client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(bytes_source=get_image_bytes(image)),
            features=[DocumentAnalysisFeature.OCR_HIGH_RESOLUTION],
        )
        result = poller.result()

        text_content: list[AzureDocumentExtractor._DocumentSection] = []

        if result.paragraphs:
            self.logger.debug("Detected %d paragraphs", len(result.paragraphs))

            # Sort all paragraphs by span's offset to read in the right order.
            result.paragraphs.sort(
                key=lambda p: (p.spans.sort(key=lambda s: s.offset), p.spans[0].offset),
            )

            for paragraph in result.paragraphs:
                match paragraph.role:
                    case None:
                        if not text_content:
                            self.logger.warning(
                                "Using %s as a title",
                                paragraph.content,
                            )
                            text_content.append(
                                AzureDocumentExtractor._DocumentSection(
                                    title=paragraph.content,
                                ),
                            )
                        else:
                            text_content[-1].paragraphs.append(paragraph.content)

                    case ParagraphRole.TITLE | ParagraphRole.SECTION_HEADING:
                        text_content.append(
                            AzureDocumentExtractor._DocumentSection(
                                title=paragraph.content,
                            ),
                        )

                    case (
                        ParagraphRole.PAGE_HEADER
                        | ParagraphRole.Page_FOOTER
                        | ParagraphRole.PAGE_NUMBER
                        | ParagraphRole.FOOTNOTE
                        | ParagraphRole.FORMULA_BLOCK
                    ):
                        self.logger.warning(
                            "Skipping %s with content %s",
                            paragraph.role,
                            paragraph.content,
                        )

                    case _:
                        self.logger.warning(
                            "Unknown paragraph role %s with content %s",
                            paragraph.role,
                            paragraph.content,
                        )

        return [section.to_article() for section in text_content]
