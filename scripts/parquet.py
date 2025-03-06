#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
# "tqdm",
# "beautifulsoup4",
# "pypandoc",
# "polars",
# ]
# ///
import logging
import random
from pathlib import Path

import polars as pl
import pypandoc  # pyright: ignore [reportMissingImports]
from bs4 import BeautifulSoup, SoupStrainer
from tqdm.contrib.concurrent import process_map

logger = logging.getLogger(__name__)

articles_dir: Path = Path("out")
if not articles_dir.exists() or not articles_dir.is_dir():
    msg = f"Directory {articles_dir} invalid"
    raise FileNotFoundError(msg)

text_dir: Path = Path("txt")
if not text_dir.exists() or not text_dir.is_dir():
    logger.warning("Converted text dir %s does not exist", text_dir)

SAMPLE: bool = False

# all files recursively
files = list(articles_dir.glob("**/*.html"))
if SAMPLE:
    files = random.sample(files, 100)

type ArticleRow = tuple[
    str, str, int, str, str, str, str | None, str | None, str | None
]

only_meta = SoupStrainer("meta")


def read_article(path: Path) -> ArticleRow:
    """Read an article from an HTML file."""
    txt: Path = text_dir / f"{path.stem}.txt"

    _, issue_page, _ = path.name.split("_", 2)
    issue, page = issue_page.split("-")
    page_no: int = int(page) + 1

    html = path.read_text()
    soup = BeautifulSoup(html, "html.parser", parse_only=only_meta)

    meta_tags = soup.find_all("meta")
    metadata: dict[str, str] = {
        tag.get("name"): tag.get("content")
        for tag in meta_tags
        if tag.has_attr("name") and tag.has_attr("content")
    }

    return (
        path.name,
        issue,
        page_no,
        metadata.get("title", "Untitled"),
        txt.read_text()
        if txt.exists()
        else pypandoc.convert_text(
            str(html), "plain", format="html", extra_args=["--wrap=none"]
        ),
        html,
        metadata.get("subtitle"),
        metadata.get("author"),
        metadata.get("category"),
    )


rows: list[ArticleRow] = process_map(
    read_article, files, desc="Reading articles", chunksize=1
)

# Convert to Polars DataFrame
articles = pl.DataFrame(
    rows,
    schema=[
        "path",
        "issue",
        "page",
        "headline",
        "text",
        "html",
        "strapline",
        "author",
        "category",
    ],
    orient="row",
)

# Save to Parquet
articles.write_parquet("felix-test.parquet")
