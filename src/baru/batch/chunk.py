import dataclasses
import functools
import json
import logging
from pathlib import Path
from typing import Self

import polars as pl
import tiktoken
from tqdm.contrib.concurrent import thread_map

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ChunkConfig:
    """Configure batch job parameters."""

    model_name: str = "text-embedding-3-large"
    """The name of the model to use."""
    endpoint: str = "/v1/embeddings"
    """The endpoint to use for requests."""
    max_batch_tasks: int = 50_000
    """The maximum number of tasks in a batch."""
    max_batch_size_bytes: int = 200 * 1024 * 1024
    """The maximum size of a batch in bytes."""
    max_batch_tokens: int = 10_000_000
    """The maximum number of tokens in a batch."""


def get_article_text(row: dict) -> str:
    """Get the text to embed."""
    content: list[str] = [row["headline"]]
    if strapline := row["strapline"]:
        content.append(strapline)
    if author := row["author"]:
        content.append(f"Author: {author}")
    content.append(row["txt"])
    return "\n\n".join(content)


@dataclasses.dataclass(frozen=True)
class BatchTask:
    """Represents a single task in a batch job."""

    id: str
    content: str
    model: str
    endpoint: str

    # Derived fields computed in __post_init__
    json: str = dataclasses.field(init=False)
    token_count: int = dataclasses.field(init=False)
    size: int = dataclasses.field(init=False)

    @classmethod
    def from_row(cls, row: dict, config: ChunkConfig) -> Self:
        """Create a BatchTask from a row of data."""
        return cls(
            id=row["filename"],
            content=(get_article_text(row)),
            model=config.model_name,
            endpoint=config.endpoint,
        )

    def __post_init__(self) -> None:
        """Compute and store derived attributes after initialisation."""
        json_str = json.dumps({
            "custom_id": self.id,
            "method": "POST",
            "url": self.endpoint,
            "body": {"model": self.model, "input": self.content},
        })
        token_count = len(tiktoken.encoding_for_model(self.model).encode(self.content))
        task_size = len(json_str.encode("utf-8") + b"\n")

        # Use object.__setattr__ for frozen dataclasses
        object.__setattr__(self, "json", json_str)
        object.__setattr__(self, "token_count", token_count)
        object.__setattr__(self, "size", task_size)


def create_batches(
    config: ChunkConfig,
    dataset_path: Path,
    output_path: Path | None = None,
) -> list[Path]:
    """Create a series of batch files for sending to the API."""
    dataset = pl.read_parquet(dataset_path)
    task_lines: list[BatchTask] = thread_map(
        functools.partial(BatchTask.from_row, config=config),
        dataset.to_dicts(),
    )

    batches: list[Path] = []

    chunk_idx = 1
    idx_start = 0
    total_tasks = len(task_lines)

    while idx_start < total_tasks:
        cur_count = 0
        cur_size = 0
        cur_tokens = 0

        # Determine the end of the current chunk
        idx_end = idx_start
        while idx_end < total_tasks:
            task = task_lines[idx_end]

            # Break out once adding the next task would exceed limits
            if (
                cur_count + 1 > config.max_batch_tasks
                or cur_size + task.size > config.max_batch_size_bytes
                or cur_tokens + task.token_count > config.max_batch_tokens
            ):
                if cur_count == 0:
                    msg = "Single task exceeds batch limits"
                    raise ValueError(msg)
                break

            cur_count += 1
            cur_size += task.size
            cur_tokens += task.token_count
            idx_end += 1

        chunk_tasks = task_lines[idx_start:idx_end]
        batch_file = (output_path or dataset_path).with_suffix(f".{chunk_idx}.jsonl")
        with batch_file.open("w", encoding="utf-8") as f:
            f.write("\n".join(task.json for task in chunk_tasks))
        logger.info(
            "Created batch %d with %d tasks (%d bytes, %d tokens)",
            chunk_idx,
            cur_count,
            cur_size,
            cur_tokens,
        )

        # Move to the next chunk
        batches.append(batch_file)
        chunk_idx += 1
        idx_start = idx_end

    return batches
