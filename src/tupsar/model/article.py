"""Dataclass storing an extracted newspaper article."""

import dataclasses
import pathlib
import re
import textwrap
import unicodedata

import bs4

formatter = bs4.formatter.HTMLFormatter(indent=0)


@dataclasses.dataclass(frozen=True)
class Article:
    """A newspaper article."""

    headline: str
    text_body: bs4.Tag
    strapline: str | None = None
    author_name: str | None = None
    category: str | None = None

    @property
    def slug(self) -> str:
        """Turn the headline into a slug."""
        return slugify(textwrap.shorten(self.headline, width=60))

    def write_out(self, output_path: pathlib.Path) -> None:
        """Save the article to an HTML file."""
        article_meta = {
            "title": self.headline.replace("\n", " "),
            "subtitle": self.strapline,
            "author": self.author_name,
            "slug": self.slug,
            "category": self.category,
        }

        # Create bs4 tags for each metadata
        meta_tags: list[bs4.Tag] = []
        for key, value in article_meta.items():
            if not value:
                continue

            meta_tags.append(
                bs4.Tag(name="meta", attrs={"name": key, "content": value})
            )

        # Package them up into the <head>
        head: bs4.Tag = bs4.Tag(name="head")
        head.extend(meta_tags)

        # Create a bs4 tag for the article body
        article_body = bs4.Tag(name="body")
        if self.text_body:
            article_body.append(self.text_body)

        # Create a bs4 tag for the entire article
        html = bs4.Tag(name="html")
        html.append(head)
        html.append(article_body)

        output_path.write_text(html.prettify(formatter=formatter), encoding="utf-8")

    @staticmethod
    def read_in(input_path: pathlib.Path) -> "Article":
        """Read an article from an HTML file."""
        soup = bs4.BeautifulSoup(input_path.read_text(encoding="utf-8"), "lxml")

        header = soup.find("head")
        body = soup.find("body")

        if not isinstance(header, bs4.Tag) or not isinstance(body, bs4.Tag):
            msg = "Deformed HTML content"
            raise TypeError(msg)

        meta_tags = header.find_all("meta")
        metadata: dict[str, str] = {
            tag.get("name"): tag.get("content")
            for tag in meta_tags
            if tag.has_attr("name") and tag.has_attr("content")
        }

        return Article(
            headline=metadata.get("title", "Untitled"),
            text_body=body,
            strapline=metadata.get("subtitle"),
            author_name=metadata.get("author"),
            category=metadata.get("category"),
        )


def slugify(text: str) -> str:
    """
    Convert a string to a hyphen-separated lowercase string.

    :param text: The string to convert
    :return: The string as a slug
    """
    # Normalise Unicode characters to their closest ASCII representation
    normalized = unicodedata.normalize("NFKD", text)
    ascii_bytes = normalized.encode("ascii", "ignore")
    ascii_str = ascii_bytes.decode("ascii")

    # Convert to lowercase
    lowercased = ascii_str.lower()

    # Replace any non-alphanumeric characters with hyphens
    hyphenated = re.sub(r"[^a-z0-9]+", "-", lowercased)

    # Remove leading and trailing hyphens
    return hyphenated.strip("-")
