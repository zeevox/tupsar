from pathlib import Path

from openai import OpenAI


def _upload_batch_job(client: OpenAI, batch_job: Path) -> str:
    # Check if the file exists
    uploaded_files = client.files.list(purpose="batch")
    for file in uploaded_files:
        if file.filename == batch_job.name and file.bytes == batch_job.stat().st_size:
            # assume that if the filename and size match, the file is the same
            return file.id

    # Upload the batch file
    with batch_job.open("rb") as f:
        file_response = client.files.create(file=f, purpose="batch")
    return file_response.id


def submit_batch_job(client: OpenAI, batch_job_file: Path) -> str:
    """Submit a batch job to the API."""
    # Sanity-check: prepared JSON lines, not parquet
    if batch_job_file.suffix != ".jsonl":
        msg = f"Invalid file path: {batch_job_file}"
        raise ValueError(msg)

    batch_job_file_id = _upload_batch_job(client, batch_job_file)

    # Create the batch job
    batch_job = client.batches.create(
        input_file_id=batch_job_file_id,
        endpoint="/v1/embeddings",
        completion_window="24h",
    )
    return batch_job.id
