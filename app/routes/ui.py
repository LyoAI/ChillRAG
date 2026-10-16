"""Web UI: a self-contained question-answering page served by the service.

Independent of the vendor-protocol routers — it adds page routes, a
list-sources endpoint, an upload path and a streaming chat endpoint, and
nothing else depends on them.
"""

import json
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app import jobs, registry
from app.auth import COOKIE, require_user
from app.corpus import attached_tool
from app.responses import ok
from chillrag.config import MATERIALS_DIR
from chillrag.corpus import CorpusBundle, CorpusManager
from chillrag.rag_tool import _folder_name, file_id_for, source_id_for
from chillrag.tools.parser import SOURCE_SUFFIXES

router = APIRouter()

_STATIC = Path(__file__).resolve().parent.parent / "static"
_PAGE = _STATIC / "index.html"
_API_PAGE = _STATIC / "api.html"
_UPLOAD_PAGE = _STATIC / "upload.html"


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_PAGE)


@router.get("/api-test", include_in_schema=False)
def api_test() -> FileResponse:
    """Fire the vendor-protocol endpoints by hand, one card per endpoint."""
    return FileResponse(_API_PAGE)


@router.get("/upload", include_in_schema=False)
def upload_page() -> FileResponse:
    """The upload page — a customer's way into a corpus, without a shell."""
    return FileResponse(_UPLOAD_PAGE)


@router.get("/space/api/service/asset-file/resource", dependencies=[Depends(require_user)])
def list_sources():
    """UI-support extension (the vendor protocol has no list endpoint)."""
    return ok([{"sourceId": sid, "resourceName": folder} for sid, folder in registry.sources()])


@router.get("/api/auth/me")
def me(user=Depends(require_user)):
    """UI-support extension: who the current ``space-auth`` cookie belongs to."""
    return ok({"username": user.username, "role": user.role})


@router.post("/api/auth/logout")
def logout() -> JSONResponse:
    """UI-support extension: expire the ``space-auth`` cookie."""
    response = JSONResponse(ok(True))
    response.delete_cookie(COOKIE)
    return response


@router.get("/api/ui/config", dependencies=[Depends(require_user)])
def ui_config():
    """UI-support extension: what the pages need that the backend already knows.

    ``suffixes`` is served rather than written into the upload page so the
    browser cannot accept a file the ingest path will then skip: ``source_docs``
    filters by suffix, so an unsupported upload would leave an empty folder, a
    job that fails, and no file to blame it on.
    """
    return ok({"suffixes": sorted(SOURCE_SUFFIXES)})


def _upload_name(filename: str) -> str:
    """The bare file name of an uploaded file.

    ``Path.name`` drops any directory part the client sent — including the
    degenerate ``.`` and ``..``, which have no name at all — so what is left to
    refuse is the empty result.
    """
    name = Path(filename).name
    if not name:
        raise HTTPException(status_code=400, detail=f"invalid file name {filename!r}")
    return name


def _resolve_target(folder: str | None, source_id: int | None) -> tuple[int, str]:
    """The ``(source id, folder)`` an upload lands in: an existing corpus or a new one."""
    if source_id is not None:
        if folder is not None:
            raise HTTPException(status_code=400, detail="give a folder name or a sourceId, not both")
        existing = registry.folder_of(source_id)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"unknown source {source_id}")
        return source_id, existing
    if folder is None:
        raise HTTPException(status_code=400, detail="give a folder name or a sourceId")
    try:
        # The check ``upload_file_complete`` and the ingest CLI both apply, so a
        # corpus a customer can name here is one a developer can ingest there.
        name = _folder_name(folder)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return source_id_for(name), name


@router.post("/api/upload", dependencies=[Depends(require_user)])
def upload(files: list[UploadFile] = [], folder: str | None = None, sourceId: int | None = None):
    """UI-support extension: a user's upload, creating the corpus if it is new.

    The vendor's ``PUT /resource`` creates a source and is admin-only; here any
    logged-in user may name a corpus, which is created on the spot. From the
    write onwards this is the vendor upload path: register, then queue the
    folder, and the client polls until the job is done.
    """
    if not files:
        raise HTTPException(status_code=400, detail="no files given")
    source_id, name = _resolve_target(folder, sourceId)

    # Every file is checked before any is written: a rejected upload must not
    # leave half a corpus behind.
    names = []
    for upload_file in files:
        file_name = _upload_name(upload_file.filename or "")
        suffix = Path(file_name).suffix.lower()
        if suffix not in SOURCE_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"{file_name}: unsupported file type {suffix or '(none)'} "
                       f"(supported: {', '.join(sorted(SOURCE_SUFFIXES))})",
            )
        names.append(file_name)

    target_dir = MATERIALS_DIR / name
    target_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for upload_file, file_name in zip(files, names):
        with (target_dir / file_name).open("wb") as sink:
            shutil.copyfileobj(upload_file.file, sink)
        written.append({
            "name": file_name,
            "fileId": file_id_for(file_name),
            "size": (target_dir / file_name).stat().st_size,
        })

    # One read-modify-write for the folder and its files, as the CLI does.
    registry.register_folder(name, names)
    jobs.submit(name)
    return ok({"sourceId": source_id, "folder": name, "files": written})


def _bundle(folder: str, state: str) -> CorpusBundle | None:
    """The folder's corpus for a status read, without ever waiting on an ingest.

    ``try_get`` answers from the loaded map or loads from disk, and gives up
    rather than waiting when the manager lock is held. That lock is what
    ``prepare`` holds across a whole parse and embed, and it is on the singleton
    rather than per folder — so ``find`` or ``get`` here would stall this poll
    for minutes whenever *any* folder is ingesting, not only this one.

    The counts are a courtesy to the upload page; ``state`` is what a client
    acts on, and it comes from the job registry, which never blocks. So a
    ``None`` bundle costs the caller its chunk counts, never its answer.
    """
    if state == "processing":
        return None

    try:
        return CorpusManager.instance().try_get(folder)
    except (OSError, ValueError):
        return None


@router.get("/api/upload/status", dependencies=[Depends(require_user)])
def upload_status(sourceId: int):
    """UI-support extension: per-file chunk counts while the upload page polls.

    The vendor's per-file status gives one code for the folder's job; a customer
    who uploaded four documents wants to see which of them landed.
    """
    folder = registry.folder_of(sourceId)
    if folder is None:
        raise HTTPException(status_code=404, detail=f"unknown source {sourceId}")
    state = jobs.status(folder)
    bundle = _bundle(folder, state)

    rows = []
    documents = chunks = 0
    for path in CorpusManager.instance().source_docs(folder):
        ingested = len(bundle.chunks_of(path.name)) if bundle else 0
        documents += 1 if ingested else 0
        chunks += ingested
        rows.append({"name": path.name, "fileId": file_id_for(path.name), "chunks": ingested})
    return ok({"state": state, "documents": documents, "chunks": chunks, "files": rows})


@router.post("/api/upload/ensure", dependencies=[Depends(require_user)])
def ensure_ingest(sourceId: int):
    """Make the source usable: register its documents, and ingest them if needed.

    The client's attach path uploads only the files the service is missing, so a
    folder whose documents are all present never reaches ``web-upload`` — and
    that route and the upload page are the only two places a job is queued. A
    folder with documents but no prepared corpus would otherwise stay that way
    forever, which is what makes this the one call that closes that gap.

    Registering is the other half. The per-file status route resolves each file
    id through the registry and silently drops the ones it does not know, so a
    folder whose documents were copied in rather than uploaded reports no status
    at all — which the client cannot tell apart from "still parsing".

    Does not wait: an ingest is minutes of parsing, so the reply reports where
    things stand and the caller polls ``/api/upload/status`` for the rest.
    """
    folder = registry.folder_of(sourceId)
    if folder is None:
        raise HTTPException(status_code=404, detail=f"unknown source {sourceId}")

    manager = CorpusManager.instance()
    registry.register_folder(folder, [p.name for p in manager.source_docs(folder)])
    # A failed job leaves no corpus behind and no job running, so it is re-queued
    # the same way an unbuilt one is; a prepared folder is left alone, or asking
    # about it would queue a redundant job behind whatever is ingesting now.
    if jobs.status(folder) == "failed" or not manager.is_prepared(folder):
        jobs.submit(folder)
    return ok({"state": jobs.status(folder), "prepared": manager.is_prepared(folder)})


class ChatBody(BaseModel):
    question: str
    source_id: int
    instructions: str | None = None
    output_instruction: str | None = None
    lang: str = ""
    history: list[dict] = []


def _payload(event) -> dict:
    return {
        "type": event.kind,
        "tool": event.tool_name,
        "args": event.tool_args,
        "text": event.text,
        "answer": event.answer,
        "num_tool_calls": event.num_tool_calls,
        "terminated": event.terminated,
        "message": event.message,
    }


@router.post("/api/chat/stream", dependencies=[Depends(require_user)])
async def chat_stream(body: ChatBody):
    """Stream one answer as JSON SSE events: tool → delta… → done/error."""
    tool = await run_in_threadpool(attached_tool, [body.source_id])

    async def events():
        try:
            async for event in tool.answer_stream(
                body.question,
                instructions=body.instructions,
                output_instruction=body.output_instruction,
                lang=body.lang,
                history=body.history,
            ):
                yield f"data: {json.dumps(_payload(event), ensure_ascii=False)}\n\n"
        except Exception as exc:  # surface as an event so the client stays in sync
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
