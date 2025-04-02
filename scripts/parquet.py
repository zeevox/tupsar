#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
# "tqdm",
# "loguru",
# "beautifulsoup4",
# "pypandoc",
# "polars",
# ]
# ///
import dataclasses
import random
from pathlib import Path
from typing import Self

import polars as pl
import pypandoc  # type: ignore[import]
from bs4 import BeautifulSoup, SoupStrainer
from loguru import logger  # type: ignore[import]
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

ARTICLES_BASE_DIR: Path = Path("out")
"""Directory containing HTML articles."""

OUTPUT_PARQUET_FILE: Path = Path("felix.parquet")
"""Output Parquet filename."""

SAMPLE_SIZE: int = 0
"""Number of articles to sample for testing. Set to 0 for all articles."""

PROCESS_CHUNK_SIZE: int = 16
"""Chunk size for parallel processing."""


@dataclasses.dataclass(frozen=True)
class Article:
    """Represents a single article."""

    filename: str
    issue: str
    page: int
    headline: str
    txt: str
    html: str
    strapline: str | None = None
    author: str | None = None
    category: str | None = None

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Read an article from an HTML file."""
        _, issue_page, _ = path.stem.split("_", 2)
        issue, page_str = issue_page.split("-")
        page_no: int = int(page_str) + 1

        html = path.read_text()
        soup = BeautifulSoup(html, "html.parser", parse_only=SoupStrainer("meta"))

        metadata: dict[str, str] = {
            tag["name"].lower(): tag["content"]  # Use lower case keys for consistency
            for tag in soup.find_all("meta", attrs={"name": True, "content": True})
        }

        return cls(
            filename=path.name,
            issue=issue,
            page=page_no,
            # Provide default values directly in .get()
            headline=metadata.get("title", "Untitled"),
            txt=pypandoc.convert_text(
                html,
                "plain",
                format="html",
                verify_format=False,
                extra_args=["--wrap=none"],
            ),
            html=html,  # Store original HTML if needed
            strapline=metadata.get("subtitle"),  # Use consistent naming if possible
            author=metadata.get("author"),
            category=metadata.get("category"),
        )


def main() -> None:
    """Convert a directory containing HTML articles into a Parquet dataset."""
    if not ARTICLES_BASE_DIR.is_dir():
        logger.error(f"Directory `{ARTICLES_BASE_DIR}` invalid or not found.")
        raise SystemExit(1)  # Exit if source directory is invalid

    logger.info("Enumerating HTML files...")
    files = list(ARTICLES_BASE_DIR.rglob("*.html"))
    logger.info(f"Found {len(files)} HTML files.")

    if not files:
        logger.warning("No HTML files found in the specified directory.")
        return  # Exit gracefully if no files are found

    if SAMPLE_SIZE:
        if len(files) > SAMPLE_SIZE:
            logger.info(f"Processing a random sample of {SAMPLE_SIZE} files.")
            files = random.sample(files, SAMPLE_SIZE)
        else:
            logger.info("Sample size >= total files. Processing all found files.")

    logger.info(f"Reading {len(files)} files...")
    results: list[Article] = process_map(
        Article.from_file,
        files,
        desc="Reading articles",
        chunksize=PROCESS_CHUNK_SIZE,
    )

    logger.info("Converting processed data to Polars DataFrame...")
    articles_df = pl.from_records([dataclasses.asdict(article) for article in results])

    # Fix incorrectly inferred column data types
    articles_df = articles_df.with_columns(
        pl.col("page").cast(pl.Int32),
    )

    logger.info(f"Saving DataFrame to Parquet file: {OUTPUT_PARQUET_FILE}")
    articles_df.write_parquet(OUTPUT_PARQUET_FILE)
    logger.info("Successfully saved data to Parquet.")


if __name__ == "__main__":
    main()
