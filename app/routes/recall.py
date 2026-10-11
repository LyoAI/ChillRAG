"""``normal_rag_recall_without_doc_group``: retrieval with all parameters in the query string.

Wire format, copied from the vendor client: ``source_ids`` is comma-joined,
``origin_ids`` is the Python repr of a list (``"[1, 2]"``), and
``recall_content_types`` repeats once per type.
"""

import ast

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import require_user
from app.corpus import attached_tool
from app.responses import ok

router = APIRouter(prefix="/space/api/service/llmapi/ht", dependencies=[Depends(require_user)])


def _parse_source_ids(raw: str) -> list[int]:
    try:
        ids = [int(part) for part in raw.split(",") if part.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail=f"bad source_ids {raw!r}") from None
    if not ids:
        # The vendor falls back to a default source here; this service has none.
        raise HTTPException(status_code=400, detail="source_ids is required")
    return ids


def _parse_origin_ids(raw: str) -> list[int]:
    if not raw.strip():
        return []
    try:
        ids = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        raise HTTPException(status_code=400, detail=f"bad origin_ids {raw!r}") from None
    if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
        raise HTTPException(status_code=400, detail="origin_ids must be a list of ints")
    return ids


@router.post("/normal_rag_recall_without_doc_group")
def recall(
    question: str,
    source_ids: str = "",
    origin_ids: str = "[]",
    context_trace_ids: str = "",
    recall_content_types: list[str] = Query(default=[]),
):
    source_id_list = _parse_source_ids(source_ids)
    tool = attached_tool(source_id_list)
    results = tool.normal_rag_recall(
        question,
        source_id_list=source_id_list,
        origin_ids=_parse_origin_ids(origin_ids),
        context_trace_ids=context_trace_ids,
        recall_content_types=recall_content_types or None,
    )
    return ok(results)
