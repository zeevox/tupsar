"""Dataclass storing an extracted newspaper article."""

import dataclasses
import pathlib
import re
import textwrap
import unicodedata

import yaml


@dataclasses.dataclass(frozen=True)
class Article:
    """A newspaper article."""

    headline: str
    text_body: str
    strapline: str | None = None
    author_name: str | None = None

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
        }

        frontmatter = yaml.dump({k: v for k, v in article_meta.items() if v})
        with output_path.open("w") as f:
            f.write(f"---\n{frontmatter}---\n\n{self.text_body}")


def slugify(text: str) -> str:
    """Convert a string to a hyphen-separated lowercase string.

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
