from pathlib import Path

import polars as pl


def merge_embeddings(dataset_path: Path, jsonl_path: Path, output_path: Path) -> None:
    """Merge downloaded embeddings into the original dataset."""
    lf_articles: pl.LazyFrame = pl.scan_parquet(dataset_path)
    has_existing_embeddings: bool = "embedding" in lf_articles.collect_schema().names()

    df_batch_output: pl.DataFrame = pl.read_ndjson(jsonl_path)
    df_embeddings: pl.DataFrame = df_batch_output.select(
        pl.col("custom_id").alias("filename"),  # Rename column for joining
        pl.col("response")
        .struct.field("body")
        .struct.field("data")
        .list.get(0)
        .struct.field("embedding")
        .alias("embedding"),
    ).filter(pl.col("filename").is_not_null() & pl.col("embedding").is_not_null())

    lf_merged: pl.LazyFrame = lf_articles.join(
        df_embeddings.lazy(),
        on="filename",
        how="left",
        suffix="_right",
        coalesce=True,
    )

    # If the dataset already had some embeddings, coalesce them
    if has_existing_embeddings:
        lf_merged = (
            lf_merged.with_columns(
                # Coalesce into embedding_new
                pl.coalesce(
                    pl.col("embedding"),
                    pl.col("embedding_right"),
                ).alias("embedding_new")
            )
            # Drop the old columns
            .drop("embedding", "embedding_right")
            # Rename the coalesced column
            .rename({"embedding_new": "embedding"})
        )

    # Write the merged DataFrame to Parquet
    lf_merged.collect().write_parquet(output_path)
