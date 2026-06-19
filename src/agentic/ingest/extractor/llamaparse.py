"""Shared LlamaParse (LlamaCloud) v2 client used by the PDF and image extractors.

Uses the v2 Parse API at the "Agentic Plus" tier (the most advanced OCR mode).
Flow: upload the file to get a ``file_id``, start a parse job with the tier, poll
until it reaches a terminal state, then fetch the per-page markdown result.
"""

import asyncio
import logging
import uuid

from agentic.ingest.extractor.base import ExtractionError

logger = logging.getLogger(__name__)

# Upload filename extension by MIME type. LlamaParse accepts PDFs and images.
_MIME_EXT = {
    "application/pdf": "pdf",
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/tiff": "tiff",
}


async def parse_pages(
    content: bytes,
    *,
    api_key: str,
    base_url: str | None,
    source_uri: str,
    extractor_name: str,
    mime_type: str = "application/pdf",
) -> list[dict]:
    """Parse *content* with LlamaParse v2 Agentic Plus.

    Returns a list of per-page dicts, each with ``page_number`` and ``markdown``.
    Raises ``ExtractionError`` on a terminal job failure, timeout, or a 4xx
    response (deterministic — the caller should not retry).
    """
    import requests

    from agentic.knowledge.model_config import (
        LLAMAPARSE_DEFAULT_BASE_URL,
        LLAMAPARSE_MAX_POLL_SECONDS,
        LLAMAPARSE_POLL_INTERVAL,
        LLAMAPARSE_TIER,
        LLAMAPARSE_TIMEOUT,
        LLAMAPARSE_VERSION,
    )

    base = base_url or LLAMAPARSE_DEFAULT_BASE_URL
    auth_headers = {"Authorization": f"Bearer {api_key}"}

    def _check(resp, step: str) -> None:
        # Any 4xx/5xx that reaches here is terminal for this call — raise
        # ExtractionError (NOT requests.HTTPError) so every caller can fall back
        # uniformly (e.g. ImageExtractor only catches ExtractionError) and the
        # whole flow is never silently retried (a retry would re-upload and start
        # another billable parse job). Includes the body for debugging.
        if resp.status_code >= 400:
            raise ExtractionError(
                f"LlamaParse {step} failed ({resp.status_code}): {resp.text[:500]}",
                extractor_name=extractor_name,
                source_uri=source_uri,
            )

    async def _get_with_retry(url, step: str, params=None):
        # Transient 5xx / network errors during polling or result-fetch are
        # retried against the SAME job_id — we NEVER re-submit a new (billable)
        # parse job to recover. Returns a response with status < 500; a 4xx is
        # left for _check to surface. Exhausting retries raises ExtractionError.
        last = None
        for attempt in range(3):
            try:
                resp = requests.get(
                    url, headers=auth_headers, params=params, timeout=LLAMAPARSE_TIMEOUT
                )
                if resp.status_code < 500:
                    return resp
                last = f"{resp.status_code}: {resp.text[:200]}"
            except requests.RequestException as e:
                last = str(e)
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
        raise ExtractionError(
            f"LlamaParse {step} failed after retries ({last})",
            extractor_name=extractor_name,
            source_uri=source_uri,
        )

    # 1. Upload the file, receive a file_id (v2 parse takes file_id, not bytes).
    # NB: the path has NO trailing slash (a trailing slash 307-redirects and
    # drops the body) and the multipart field is "upload_file". Use a UNIQUE
    # upload name: LlamaParse upserts its file record by name, so a constant name
    # makes concurrent extractions share one record and clobber each other.
    ext = _MIME_EXT.get(mime_type, "bin")
    upload_name = f"agentic-{uuid.uuid4().hex}.{ext}"
    upload_resp = requests.post(
        f"{base}/api/v1/files",
        headers=auth_headers,
        files={"upload_file": (upload_name, content, mime_type)},
        timeout=LLAMAPARSE_TIMEOUT,
    )
    _check(upload_resp, "file upload")
    file_id = upload_resp.json()["id"]

    # 2. Start a v2 parse job at the Agentic Plus tier.
    parse_resp = requests.post(
        f"{base}/api/v2/parse",
        headers={**auth_headers, "Content-Type": "application/json"},
        json={"file_id": file_id, "tier": LLAMAPARSE_TIER, "version": LLAMAPARSE_VERSION},
        timeout=LLAMAPARSE_TIMEOUT,
    )
    _check(parse_resp, "parse start")
    job_id = parse_resp.json()["id"]
    logger.info(
        "LlamaParse: parsing file %s (tier=%s), polling job %s",
        file_id, LLAMAPARSE_TIER, job_id,
    )

    # 3. Poll until the job reaches a terminal state.
    job_url = f"{base}/api/v2/parse/{job_id}"
    terminal_failures = ("FAILED", "CANCELLED", "ERROR")
    in_progress = ("PENDING", "RUNNING", "PROCESSING", "QUEUED", "STARTED")
    warned_statuses: set[str] = set()
    elapsed = 0
    while True:
        status_resp = await _get_with_retry(job_url, "job poll")
        _check(status_resp, "job poll")
        status_body = status_resp.json()
        # v2 nests the job state under "job" on the GET; the parse-start response
        # uses a top-level "status" — accept either.
        status = (status_body.get("job") or {}).get("status") or status_body.get("status")
        if status == "COMPLETED":
            break
        if status in terminal_failures:
            raise ExtractionError(
                f"LlamaParse job {job_id} failed with status {status}",
                extractor_name=extractor_name,
                source_uri=source_uri,
            )
        # Don't silently wait out the full timeout on an unrecognized status
        # (typo'd / new API value) — surface it so the cause isn't masked.
        if status not in in_progress and status not in warned_statuses:
            logger.warning(
                "LlamaParse job %s returned unrecognized status %r; still polling",
                job_id, status,
            )
            warned_statuses.add(status)
        if elapsed >= LLAMAPARSE_MAX_POLL_SECONDS:
            raise ExtractionError(
                f"LlamaParse job {job_id} did not complete within "
                f"{LLAMAPARSE_MAX_POLL_SECONDS}s (last status: {status})",
                extractor_name=extractor_name,
                source_uri=source_uri,
            )
        await asyncio.sleep(LLAMAPARSE_POLL_INTERVAL)
        elapsed += LLAMAPARSE_POLL_INTERVAL

    # 4. Fetch the per-page markdown result.
    result_resp = await _get_with_retry(job_url, "result fetch", params={"expand": "markdown"})
    _check(result_resp, "result fetch")
    markdown = result_resp.json().get("markdown")
    if isinstance(markdown, dict):
        raw_pages = markdown.get("pages") or []
    elif isinstance(markdown, list):
        raw_pages = markdown
    else:
        # markdown returned as a single string (or absent) — treat as one page.
        raw_pages = [{"markdown": markdown or ""}]

    pages = []
    for i, page in enumerate(raw_pages):
        page_md = page.get("markdown") or page.get("md") or page.get("text") or ""
        page_num = page.get("page") or page.get("page_number") or i + 1
        pages.append({"page_number": page_num, "markdown": page_md})
    return pages
