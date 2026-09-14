"""Source-media ingestion for media tasks.

Media chaining references prior jobs by id, which covers art Conduct itself
generated. External art — an existing book cover headed for upscale_image /
inpaint_image, say — needs a way in: an authenticated client POSTs the file
here, it lands under `{output_dir}/uploads/`, and the returned `source_url`
is what `inputs.source_image_url` accepts on a media job. The worker reads
it from the shared output volume, so the source never crosses the HTTP
boundary again after this one hop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel

from config.settings import get_settings
from models.client import ClientApp
from rate_limit import rate_limited_client

router = APIRouter(prefix="/uploads", tags=["uploads"])

_ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
# Print art runs big — a 300-DPI full-wrap cover PNG is tens of MB — but
# nothing legitimate needs more than this.
_MAX_BYTES = 64 * 1024 * 1024


class UploadOut(BaseModel):
    upload_id: str
    source_url: str
    bytes: int


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_media(
    file: UploadFile,
    client: Annotated[ClientApp, Depends(rate_limited_client)],
) -> UploadOut:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported file type {suffix or '(none)'} — "
            f"allowed: {', '.join(sorted(_ALLOWED_SUFFIXES))}",
        )
    data = await file.read(_MAX_BYTES + 1)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty upload")
    if len(data) > _MAX_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"upload exceeds {_MAX_BYTES // (1024 * 1024)}MB limit",
        )

    upload_id = uuid4()
    # The client's filename never touches the filesystem — only its suffix
    # survives, on a server-minted UUID name.
    name = f"{upload_id}{suffix}"
    out_dir = Path(get_settings().tts_output_dir) / "uploads"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_bytes(data)

    return UploadOut(
        upload_id=str(upload_id),
        source_url=f"/output/uploads/{name}",
        bytes=len(data),
    )
