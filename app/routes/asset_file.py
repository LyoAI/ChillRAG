"""Source (folder), upload, status and segment endpoints.

Mirrors the vendor's ``asset-file`` group. A source is a folder under
``CHILLRAG_MATERIALS_DIR``; an upload writes into it and queues an ingest job.
"""

import hashlib
import json
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel

from app import jobs, registry
from app.auth import require_admin, require_user
from app.corpus import attached_tool
from app.responses import ok
from chillrag.config import MATERIALS_DIR
from chillrag.corpus import CorpusManager
from chillrag.rag_tool import file_id_for, source_id_for

router = APIRouter(prefix="/space/api/service/asset-file", dependencies=[Depends(require_user)])


class ResourceBody(BaseModel):
    resourceName: str
    userId: int | None = None


def _safe_name(name: str) -> str:
    """A bare file or folder name: no separators, no parent references."""
    if name in ("", ".", "..") or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail=f"invalid name {name!r}")
    return name


def _digest(path: Path) -> str | None:
    """The file's SHA-256, or ``None`` when it is not there."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@router.put("/resource", dependencies=[Depends(require_admin)])
def create_source(body: ResourceBody):
    folder = _safe_name(body.resourceName)
    (MATERIALS_DIR / folder).mkdir(parents=True, exist_ok=True)
    source_id = source_id_for(folder)
    registry.register_source(source_id, folder)
    return ok(source_id)


@router.post("/web-upload")
def upload(sourceId: int, files: list[UploadFile]):
    """Write the upload and queue an ingest — but only if it changed anything.

    An upload of files the corpus already has is not free to accept. ``submit``
    flips the folder's status to ``processing`` the moment it is called and puts
    the job behind whatever the pool's single worker is parsing, and the
    per-file status route answers from that registry. So re-uploading a corpus
    that is ready makes it *report* as un-ready until the queue ahead drains —
    which is minutes to hours on a real corpus, and which the client reads as
    "still parsing" and gives up on. The corpus itself is untouched either way.

    The client re-uploads when its status query fails, and that query fails
    while *any* folder is parsing, because the manager lock is global. So a
    corpus being parsed elsewhere was enough to put this one back in the queue,
    which queued another job, and so on.

    Both conditions below are cheap and take no lock, so they hold while another
    folder is parsing — which is exactly when this matters. Byte-identical
    content means the corpus is not missing anything this upload brought; an
    unprepared folder means there is nothing to keep it in step with. What this
    cannot see is a document that was copied into the folder by hand and never
    ingested: it is already on disk, so no upload brings it either.
    """
    folder = registry.folder_of(sourceId)
    if folder is None:
        raise HTTPException(status_code=404, detail=f"unknown source {sourceId}")
    target_dir = MATERIALS_DIR / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    file_ids = []
    changed = False
    for upload_file in files:
        name = _safe_name(Path(upload_file.filename or "").name)
        target = target_dir / name
        before = _digest(target)
        with target.open("wb") as sink:
            shutil.copyfileobj(upload_file.file, sink)
        # Once one file differs the answer cannot change back, and the digest
        # reads the whole file — so stop paying for it.
        changed = changed or _digest(target) != before
        file_id = file_id_for(name)
        registry.register_file(file_id, sourceId, name)
        file_ids.append(file_id)
    if changed or not CorpusManager.instance().is_prepared(folder):
        jobs.submit(folder)
    return ok(file_ids)


@router.post("")
def file_status(file_ids: list[int]):
    """Per-file status; ``code`` is ``finish`` once its folder's ingest is done."""
    rows = []
    for file_id in file_ids:
        record = registry.file_of(file_id)
        if record is None:
            continue
        folder = registry.folder_of(record["source"])
        code = jobs.status(folder) if folder else "failed"
        rows.append({
            "id": file_id,
            "fileName": record["name"],
            "status": [{"code": code}],
        })
    return ok(rows)


@router.get("/segments/{file_id}")
def segments(file_id: int):
    """Whole-document readback. ``segment.content`` is a JSON string holding
    ``{"content": text}``: the vendor client ``json.loads`` it without a fallback."""
    folder = registry.folder_of_file(file_id)
    if folder is None:
        raise HTTPException(status_code=404, detail=f"unknown file {file_id}")
    tool = attached_tool([registry.file_of(file_id)["source"]])
    rows = tool.get_segments(file_id)
    return ok([
        {"segment": {
            "index": row["segment"]["index"],
            "content": json.dumps({"content": row["segment"]["content"]}, ensure_ascii=False),
        }}
        for row in rows
    ])
