"""``post-answer``: answer a question with the agent, streamed as the vendor's SSE.

The vendor client reads every line, strips an optional ``data:`` prefix, turns
``[new_line]`` and ``[whitespace]`` back into characters, then keeps the text
after the first ``</strong>``. The preamble here is therefore an empty
``<strong></strong>``: the client drops it and keeps the answer.
"""

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.auth import require_user
from app.corpus import attached_tool, folders_of
from app.responses import fail

router = APIRouter(prefix="/space/api/service/sse/llmapi/ht/text", dependencies=[Depends(require_user)])

PREAMBLE = "<strong></strong>"


class AnswerBody(BaseModel):
    question: str
    source_category_filter: str = "{}"  # JSON object keyed by source id, as the vendor sends it
    mode: int = 0  # accepted for compatibility; the agent has one mode


def encode(text: str) -> str:
    return text.replace("\n", "[new_line]").replace(" ", "[whitespace]")


def _stream(answer: str) -> Iterator[str]:
    # One event: the client joins data lines without separators, so splitting on
    # newlines here would silently drop them. Newlines travel as [new_line].
    yield f"data: {encode(PREAMBLE + answer)}\n\n"


@router.post("/post-answer")
def post_answer(body: AnswerBody):
    try:
        source_ids = [int(key) for key in json.loads(body.source_category_filter)]
    except (ValueError, TypeError):
        return fail(400, "source_category_filter must be a JSON object keyed by source id")
    # One corpus per answer: a RAGTool holds one attached folder, so a second
    # would be dropped silently and the answer drawn from the wrong documents.
    if len(folders_of(source_ids)) != 1:
        return fail(400, "post-answer takes exactly one source")
    tool = attached_tool(source_ids)
    try:
        answer = tool.answer_by_question(body.question, source_id_list=source_ids)
    except TimeoutError:
        return fail(504, "answer timed out")
    return StreamingResponse(_stream(answer), media_type="text/event-stream")
