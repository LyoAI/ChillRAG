"""Web UI: a self-contained question-answering page served by the service.

Independent of the vendor-protocol routers — it adds a page route, a
list-sources endpoint and a streaming chat endpoint, and nothing else depends
on them.
"""

import json
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app import registry
from app.auth import COOKIE, require_user
from app.corpus import attached_tool
from app.responses import ok

router = APIRouter()

_PAGE = Path(__file__).resolve().parent.parent / "static" / "index.html"


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_PAGE)


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
