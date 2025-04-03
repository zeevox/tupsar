from typing import Literal

import diff_match_patch
import rich.text
from unidecode import unidecode

COLLAPSE_THRESHOLD = 5


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


def diff_to_rich(
    diffs: list[tuple[Literal[-1, 0, 1], str]], *, collapse: bool = True
) -> rich.text.Text:
    """Convert a diff-match-patch diff to a rich text object."""
    rich_text = rich.text.Text()

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
        elif op == 1:  # Insertion.
            rich_text.append(segment, style="bold green")
        elif op == -1:  # Deletion.
            rich_text.append(segment, style="bold strike red")
        else:
            msg = f"Unexpected diff operation: {op}"
            raise ValueError(msg)

        ended_newline = segment.endswith("\n")

    return rich_text
