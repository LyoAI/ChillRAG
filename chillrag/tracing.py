"""Agent episode tracing — one JSONL file per session, under ``TRACE_DIR``.

Layout: ``<TRACE_DIR>/<user>/<session>.jsonl``, one line per agent episode.
A record is complete and self-contained — the exact wire transcript
(``messages``: system / user / assistant+tool_calls / tool) plus episode
scalars — so one line reconstructs one training sample, an audit trail
entry, and the context needed to resume the session. Nothing derived from
the transcript is stored alongside it (no per-call trajectory view): the
transcript is the record.

Recording is automatic: the agent loop writes a record when an episode ends.
What it needs from the caller is only the request identity — who is asking,
which session, from which surface (``source``: the web app or the vendor
API) — which the caller binds with :func:`episode` around the answering
call. Outside a bound context (library use, scripts) recording is a no-op:
an episode with no user behind it has nothing to be attributed to.

Every channel supplies an identity through this binding: the web app under
the page's session id (``source: "web"``), the vendor ``post-answer`` under a
freshly minted per-call id (``source: "api"`` — the protocol has no session
field, but every record should still carry a real session). An explicit
``None`` session falls back to the user's ``_adhoc`` file — the safety net
for callers that trace but name no session.

Compaction: a session file past ``TRACE_COMPACT_BYTES`` is rewritten with
the newest record verbatim and every older record reduced to its scalar
fields (``"compacted": true``, no ``messages``). Resuming a session only
needs the newest record — its messages hold the whole conversation — so
compaction never costs resumability; what it costs is older episodes'
tool-result payloads, irreversibly.

Failures are non-fatal by design: a trace that cannot be written logs a
warning and never fails the answer that produced it.
"""

from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from chillrag.config import TRACE_COMPACT_BYTES, TRACE_DIR
from chillrag.utils import get_logger

logger = get_logger()

#: Session (and user) names are filenames here, and a client supplies the
#: session — so the charset is enforced, not trusted. Same family as auth's
#: username rule.
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: File name for episodes bound with an explicit ``None`` session — the safety
#: net for callers that trace but have no session to name.
ADHOC_SESSION = "_adhoc"

#: The request identity of the episode being answered, if any. Copies into
#: threads (``asyncio.to_thread``, ``run_in_threadpool``) and nested event
#: loops (``asyncio.run``) automatically — that propagation is what makes
#: the recording invisible to the answer path.
_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar("chillrag_trace", default=None)

#: Mutable slot holding the id of the episode most recently recorded under the
#: current binding. A dict rather than a ``ContextVar[str]``: the agent records
#: inside ``asyncio.run``, whose context copy would swallow a rebinding, while a
#: mutation of this shared object is visible back in the caller — that is how a
#: driver like ``batch_request`` learns the id of the trace it just caused.
_LAST_RECORD: ContextVar[dict[str, Any] | None] = ContextVar(
    "chillrag_trace_last", default=None
)


def _validate_name(name: str, kind: str) -> str:
    if not _NAME_RE.match(str(name)):
        raise ValueError(f"invalid trace {kind} {name!r}")
    return str(name)


def new_session_id() -> str:
    """A fresh session id for a channel that has none.

    Every channel's records should carry a real session id, so an entry point
    whose protocol has no session (the vendor ``post-answer``) mints one per
    call here instead of letting records fall into ``_adhoc``.
    """
    return uuid4().hex[:12]


def resolve_session(session_id: str | None) -> str:
    """The file stem a session's records append to.

    ``None`` (no session given) resolves to ``_adhoc``. An invalid id raises:
    refusing beats mangling, since the caller may resume from this file later
    and a silent rename would break the linkage.
    """
    if not session_id:
        return ADHOC_SESSION
    if not _NAME_RE.match(session_id):
        raise ValueError(
            f"invalid session_id {session_id!r}: use letters, digits, _ or -, "
            f"at most 64 characters"
        )
    return session_id


@contextmanager
def episode(*, user: str, session_id: str | None = None, **fields: Any) -> Iterator[None]:
    """Bind the request identity for the agent episodes answered inside the block.

    This is the only thing a caller supplies. The agent records on its own
    when an episode ends; it cannot know who asked, so the identity travels
    here instead of through any answer-path signature. Names are validated on
    entry — a bad one fails the request at its boundary rather than silently
    losing its traces later.
    """
    fields["user"] = _validate_name(user, "user")
    fields["session_id"] = resolve_session(session_id)
    slot: dict[str, Any] = {}
    token = _CONTEXT.set(fields)
    slot_token = _LAST_RECORD.set(slot)
    try:
        yield
    finally:
        _LAST_RECORD.reset(slot_token)
        _CONTEXT.reset(token)


def last_trace_id() -> str | None:
    """The id of the episode most recently recorded under this binding.

    ``None`` outside an :func:`episode` block, and also when nothing was
    recorded — an answer served from the cache never reaches the agent, so
    there is no trace to name.
    """
    slot = _LAST_RECORD.get()
    return slot.get("trace_id") if slot else None


def record_episode(result: Any) -> None:
    """Write one record for a finished episode; the agent calls this itself.

    A no-op outside a bound :func:`episode` context. Never raises (see the
    module docstring).
    """
    context = _CONTEXT.get()
    if context is None:
        return
    trace_id = uuid4().hex
    written = TraceLog.instance().write({
        "trace_id": trace_id,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **context,
        # The agent knows what it was asked; the identity only supplies who
        # and from where. Taken from the result, so one episode() block can
        # wrap a whole batch of differently-phrased questions.
        "question": result.question,
        "answer": result.answer,
        "num_tool_calls": result.num_tool_calls,
        "terminated": result.terminated,
        "total_tokens": result.total_tokens,
        "duration_s": round(result.duration_s, 3),
        # Last: the bulk of the line, and the part every consumer
        # (training, resume, audit) actually reads around.
        "messages": result.messages,
    })
    if written:
        # Only a record actually on disk gets to be named; reporting the id of
        # a failed write would send a driver looking for a trace that is not
        # there.
        slot = _LAST_RECORD.get()
        if slot is not None:
            slot["trace_id"] = trace_id


class TraceLog:
    """Appends episode records to per-session JSONL files.

    Use ``TraceLog.instance()`` for the shared one (rooted at ``TRACE_DIR``);
    constructing a separate instance with an explicit root is for tests and
    one-off tooling.
    """

    _instance: "TraceLog | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "TraceLog":
        """The process-wide trace log."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self, trace_dir: str | Path | None = None,
                 compact_bytes: int | None = None) -> None:
        self.trace_dir = Path(trace_dir) if trace_dir else TRACE_DIR
        self.compact_bytes = compact_bytes if compact_bytes is not None else TRACE_COMPACT_BYTES
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> bool:
        """Append one record as a JSON line; never raises.

        Compaction runs behind the same lock, so a size read always sees the
        append it follows.

        Returns:
            Whether the line reached the file.
        """
        with self._lock:
            try:
                path = self.session_path(record["user"], record["session_id"])
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._compact_if_needed(path)
                return True
            except (OSError, ValueError) as e:
                logger.warning(
                    "trace not written for %s/%s: %s",
                    record.get("user"), record.get("session_id"), e,
                )
                return False

    def session_path(self, user: str, session_id: str | None) -> Path:
        """``<trace_dir>/<user>/<session>.jsonl`` for a record's identity."""
        user = _validate_name(user, "user")
        session = resolve_session(session_id)
        return self.trace_dir / user / f"{session}.jsonl"

    # ---------------------------------------------------------- compact --

    def _compact_if_needed(self, path: Path) -> None:
        """Rewrite an oversized session file: newest record verbatim, the rest
        reduced to scalars. A single-record file has nothing to drop."""
        if path.stat().st_size <= self.compact_bytes:
            return
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) < 2:
            logger.warning(
                "session %s exceeds the compact threshold on one record; left as-is",
                path,
            )
            return
        for old in records[:-1]:
            old.pop("messages", None)
            old["compacted"] = True

        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        size = tmp.stat().st_size
        tmp.replace(path)
        logger.info(
            "compacted %s: %d older records reduced to scalars, %d bytes kept verbatim",
            path, len(records) - 1, size,
        )
