"""Dataclass storing an extracted newspaper article."""

import dataclasses
import pathlib
import re
import textwrap
import unicodedata


@dataclasses.dataclass(frozen=True)
class Article:
    """A newspaper article."""

    issue: str
    page_no: int
    headline: str
    text_body: str
    strapline: str | None = None
    author_name: str | None = None
    category: str | None = None

    @property
    def slug(self) -> str:
        """Turn the headline into a slug."""
        return slugify(textwrap.shorten(self.headline, width=60))

    def write_out(self, output_path: pathlib.Path) -> None:
        """Save the article to a Markdown file."""
        article_meta = {
            "title": self.headline.replace("\n", " "),
            "subtitle": self.strapline,
            "author": self.author_name,
            "slug": self.slug,
            "category": self.category,
            "issue": self.issue,
            "page_no": self.page_no,
        }

        meta_tags = "\n    ".join(
            f'<meta name="{k}" content="{v}">' for k, v in article_meta.items() if v
        )
        html = f"""<!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          {meta_tags}
        </head>
        <body>
          {self.text_body}
        </body>
        </html>
        """
        output_path.write_text(html, encoding="utf-8")


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
