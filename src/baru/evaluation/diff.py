from typing import Literal

import diff_match_patch
from rich.style import Style
from rich.text import Text
from unidecode import unidecode

type DmpDiff = tuple[Literal[-1, 0, 1], str]

COLLAPSE_THRESHOLD = 5

INSERTION_STYLE = Style(color="bright_green", bgcolor="dark_green", bold=True)
DELETION_STYLE = Style(color="bright_red", bgcolor="dark_red", bold=True)

OP_STYLE_MAP: dict[int, Style] = {
    1: INSERTION_STYLE,
    -1: DELETION_STYLE,
}


def compute_diff(
    text1: str, text2: str, edit_cost: int = 8, timeout: float = 0.0
) -> list:
    """Diff two texts."""
    # Normalize the texts to remove any Unicode characters
    text1 = unidecode(text1, errors="strict")
    text2 = unidecode(text2, errors="strict")

    dmp = diff_match_patch.diff_match_patch()
    dmp.Diff_EditCost = edit_cost
    dmp.Diff_Timeout = timeout

    diffs = dmp.diff_main(text1, text2, checklines=False)

    # cleanup to remove overzealous coincidences
    dmp.diff_cleanupEfficiency(diffs)

    return diffs


def diff_to_rich(diffs: list[DmpDiff], *, collapse: bool = True) -> Text:
    """Convert a diff-match-patch diff to a rich text object."""
    rich_text = Text()

    ended_newline = True
    for op, segment in diffs:
        if op == 0:  # Equal text.
            if collapse and segment.count("\n") >= COLLAPSE_THRESHOLD:
                skipped_lines: list[str] = segment.splitlines(keepends=True)
                if not ended_newline:
                    rich_text.append(skipped_lines.pop(0))
                last: str | None = (
                    skipped_lines.pop(-1)
                    if not skipped_lines[-1].endswith("\n")
                    else None
                )
                summary = f"\n[{len(skipped_lines)} unchanged lines]\n\n"
                rich_text.append(summary, style="dim")
                if last is not None:
                    rich_text.append(last)
            else:
                rich_text.append(segment)
        else:
            # replace newlines with U+23CE ⏎ RETURN SYMBOL
            rich_text.append(segment.replace("\n", "\u23ce\n"), OP_STYLE_MAP.get(op))

        ended_newline = segment.endswith("\n")

    return rich_text
