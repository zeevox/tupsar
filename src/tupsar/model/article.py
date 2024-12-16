"""Dataclass storing an extracted newspaper article."""

import dataclasses
import pathlib

import yaml


@dataclasses.dataclass(frozen=True)
class Article:
    """A newspaper article."""

    headline: str
    strapline: str | None
    author_name: str | None
    section: str
    text_body: str
    slug: str

    @staticmethod
    def from_json(
        article: dict[str, str],
    ) -> None:
        """Initialise the article from a JSON dictionary."""
        return Article(
            headline=article.get("headline"),
            strapline=article.get("strapline"),
            author_name=article.get("author_name"),
            section=article.get("section"),
            text_body=article.get("text_body"),
            slug=article.get("slug"),
        )

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
