"""Dataclass storing an extracted newspaper article."""

import dataclasses
import pathlib

import yaml


@dataclasses.dataclass(frozen=True)
class Article:
    """A newspaper article."""

    headline: str
    section: str
    text_body: str
    slug: str
    strapline: str | None = None
    author_name: str | None = None

    def write_out(self, output_path: pathlib.Path) -> None:
        """Save the article to a Markdown file."""
        article_meta = {
            "title": self.headline,
            "subtitle": self.strapline,
            "author": self.author_name,
            "slug": self.slug,
            "category": self.section,
        }

        frontmatter = yaml.dump({k: v for k, v in article_meta.items() if v})
        with output_path.open("w") as f:
            f.write(f"---\n{frontmatter}---\n\n{self.text_body}")
