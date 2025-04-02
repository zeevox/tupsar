#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
# "tqdm",
# "natsort",
# "rich",
# ]
# ///

import argparse
from pathlib import Path

import natsort
import rich


def find_missing_scans(scans_dir: Path, articles_dir: Path) -> list[Path]:
    """Find unprocessed scans lacking any extracted articles."""
    articles = {
        # split on second underscore
        "_".join(article.split("_")[:2])
        for _, _, files in articles_dir.walk()
        for article in files
    }

    missing_scans = {scan for scan in scans_dir.iterdir() if scan.stem not in articles}

    return natsort.natsorted(missing_scans)


def cli() -> None:
    """Handle command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("scans_dir", type=str)
    parser.add_argument("articles_dir", type=str)
    args = parser.parse_args()

    scans = Path(args.scans_dir)
    articles = Path(args.articles_dir)

    for folder in [scans, articles]:
        if not folder.exists():
            msg = f"Directory {folder} does not exist"
            raise FileNotFoundError(msg)

    for scan in find_missing_scans(scans, articles):
        rich.print(scan)


if __name__ == "__main__":
    cli()
