"""Extract articles from newspaper scans using Gemini."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from tupsar.extractor.gemini import GeminiExtractor


def main() -> None:
    if not load_dotenv():
        raise FileNotFoundError

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to file or directory")
    parser.add_argument("-o", "--output", type=str, help="Output directory")
    args = parser.parse_args()

    input_path = args.path
    output_dir = args.output

    # derive output path from input image path
    output_path = (
        Path(output_dir) if output_dir else Path("output") / Path(input_path).stem
    )
    output_path.mkdir(exist_ok=True, parents=True)

    articles = GeminiExtractor().extract(Path(input_path))

    # Print article text
    for idx, article in enumerate(articles):
        article_path = output_path / f"{idx}_{article.slug}.md"
        article.write_out(article_path)
