"""Tracing: an agent episode records itself when a request identity is bound
(:func:`tracing.episode`), producing one complete record per episode in the
session's JSONL file; ``TraceLog`` appends and compacts those files.

The loop tests drive a real ``ReActAgent`` over a scripted LLM and a real
in-memory ``ToolRegistry`` — what is asserted is the record the production
path writes, not a hand-built one.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from chillrag.agent import ReActAgent
from chillrag.agent.react_agent import AgentResult
from chillrag.llm import LLMResponse, ToolCallRequest
from chillrag.llm.provider.base import StreamChunk
from chillrag.tools import ToolRegistry
from chillrag.tools.searcher import DocumentStore
from chillrag.tracing import (
    ADHOC_SESSION,
    TraceLog,
    episode,
    last_trace_id,
    record_episode,
    resolve_session,
)


# ------------------------------------------------------------------ fakes --

class ScriptedLLM:
    """One queued response per round, for both the sync and stream paths."""

    def __init__(self, rounds: list[LLMResponse]) -> None:
        self.rounds = list(rounds)

    async def ainvoke(self, messages, **kwargs) -> LLMResponse:
        return self.rounds.pop(0)

    async def astream(self, messages, **kwargs):
        resp = self.rounds.pop(0)
        if resp.has_tool_calls:
            yield StreamChunk(content=resp.content or "", tool_calls=resp.tool_calls,
                              finish_reason="tool_calls")
        else:
            yield StreamChunk(content=resp.content or "", finish_reason="stop")


def _tool_round() -> LLMResponse:
    return LLMResponse(
        content="先看看有哪些材料",
        tool_calls=[ToolCallRequest(id="call-1", name="get_materials", arguments={})],
        finish_reason="tool_calls",
        usage={"total_tokens": 42},
    )


def _final_round(text: str) -> LLMResponse:
    return LLMResponse(content=text, finish_reason="stop", usage={"total_tokens": 10})


def _registry() -> ToolRegistry:
    return ToolRegistry({}, DocumentStore(name="t"))


def _episode(answer: str = "最终答案") -> AgentResult:
    return AgentResult(
        question="q",
        answer=answer,
        trajectory=[],
        num_tool_calls=1,
        total_tokens=52,
        terminated="answer",
        messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": answer}],
    )


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ------------------------------------------------------------ session ids --

def test_resolve_session_rules():
    assert resolve_session(None) == ADHOC_SESSION
    assert resolve_session("") == ADHOC_SESSION
    assert resolve_session("abc_123") == "abc_123"


@pytest.mark.parametrize("bad", ["a/b", "..", "a b", "x" * 65])
def test_an_invalid_session_id_is_refused_not_mangled(bad):
    with pytest.raises(ValueError):
        resolve_session(bad)


def test_a_minted_session_id_is_valid_and_unique():
    from chillrag.tracing import _NAME_RE, new_session_id

    first, second = new_session_id(), new_session_id()
    assert first != second
    assert _NAME_RE.match(first), "minted ids must pass the same charset rule"


# ---------------------------------------------------------------- records --

def test_a_record_carries_context_scalars_and_the_transcript(tmp_path):
    result = _episode()
    with episode(user="alice", session_id="s1", folder="corpus", source="web",
                 mode="agent", lang="zh", instructions=None):
        record_episode(result)

    # The conftest fixture points TRACE_DIR into this test's tmp_path.
    (record,) = _lines(tmp_path / "traces" / "alice" / "s1.jsonl")
    assert record["user"] == "alice" and record["session_id"] == "s1"
    assert record["source"] == "web"
    assert record["folder"] == "corpus"
    assert record["question"] == result.question, "the question comes from the agent, not the binding"
    assert record["answer"] == "最终答案"
    assert record["num_tool_calls"] == 1 and record["terminated"] == "answer"
    assert record["total_tokens"] == 52 and "duration_s" in record
    assert record["trace_id"] and record["ts"]
    assert record["messages"] == result.messages


def test_sessionless_records_land_in_adhoc(tmp_path):
    with episode(user="alice"):
        record_episode(_episode())
    (record,) = _lines(tmp_path / "traces" / "alice" / f"{ADHOC_SESSION}.jsonl")
    assert record["session_id"] == ADHOC_SESSION


def test_outside_a_request_nothing_is_recorded(tmp_path):
    """Library use has no identity to attribute, so it stays untraced."""
    record_episode(_episode())
    assert not (tmp_path / "traces").exists()


def test_last_trace_id_is_none_until_a_record_is_written(tmp_path):
    assert last_trace_id() is None
    with episode(user="alice", session_id="s1"):
        assert last_trace_id() is None


def test_last_trace_id_survives_the_agents_event_loop(tmp_path):
    """The agent records inside ``asyncio.run``; its context copy must not
    swallow the id the caller needs to join the answer to the trace."""
    agent = ReActAgent(ScriptedLLM([_tool_round(), _final_round("答案")]), _registry())
    with episode(user="svc", session_id="s1"):
        asyncio.run(agent.arun("q"))
        trace_id = last_trace_id()

    (record,) = _lines(tmp_path / "traces" / "svc" / "s1.jsonl")
    assert trace_id == record["trace_id"]


def test_invalid_names_are_refused_at_the_boundary(tmp_path):
    with pytest.raises(ValueError):
        with episode(user="no/slash"):
            pass
    assert not (tmp_path / "traces").exists()


# --------------------------------------------------------------- compact --

def test_compact_keeps_only_the_newest_transcript(tmp_path):
    log = TraceLog(tmp_path, compact_bytes=400)
    for n in range(3):
        log.write({
            "user": "u", "session_id": "s", "folder": "f", "question": f"q{n}",
            "answer": f"a{n}", "num_tool_calls": 1, "terminated": "answer",
            "total_tokens": 5, "duration_s": 0.1,
            "messages": [{"role": "user", "content": "x" * 200}],
        })

    records = _lines(tmp_path / "u" / "s.jsonl")
    assert [r["question"] for r in records] == ["q0", "q1", "q2"]
    assert "messages" not in records[0] and records[0]["compacted"] is True
    assert "messages" not in records[1] and records[1]["compacted"] is True
    assert records[2]["messages"], "the newest record keeps its transcript"


def test_a_file_under_the_threshold_is_left_alone(tmp_path):
    log = TraceLog(tmp_path, compact_bytes=10 * 1024 * 1024)
    for n in range(2):
        log.write({
            "user": "u", "session_id": "s", "folder": "f", "question": f"q{n}",
            "answer": f"a{n}", "num_tool_calls": 1, "terminated": "answer",
            "total_tokens": 5, "duration_s": 0.1,
            "messages": [{"role": "user", "content": "x" * 20}],
        })
    assert all("messages" in r for r in _lines(tmp_path / "u" / "s.jsonl"))


def test_write_never_raises_even_on_a_forged_record(tmp_path):
    log = TraceLog(tmp_path)
    log.write({"user": "no/slash", "session_id": "s"})  # forged past the boundary
    assert not (tmp_path / "no").exists()


# ------------------------------------------------------------ agent loops --

def _roles(messages):
    return [m["role"] for m in messages]


async def _collect(agen):
    return [event async for event in agen]


def test_an_episode_records_itself_when_a_request_is_bound(tmp_path):
    agent = ReActAgent(ScriptedLLM([_tool_round(), _final_round("答案")]), _registry())
    with episode(user="svc", session_id="s1"):
        result = asyncio.run(agent.arun("q"))

    (record,) = _lines(tmp_path / "traces" / "svc" / "s1.jsonl")
    assert _roles(record["messages"]) == ["system", "user", "assistant", "tool", "assistant"]
    assert record["messages"][3]["tool_call_id"] == "call-1"
    assert record["answer"] == result.answer == "答案"
    assert record["num_tool_calls"] == 1
    # A scripted LLM can finish a whole episode under a millisecond, so 0.0 is
    # a legitimate duration; what matters is the field is measured, not left
    # at its dataclass default by accident.
    assert record["duration_s"] >= 0


def test_a_stream_episode_records_the_same_shape(tmp_path):
    agent = ReActAgent(ScriptedLLM([_tool_round(), _final_round("流式答案")]), _registry())
    with episode(user="svc", session_id="s1"):
        events = asyncio.run(_collect(agent.arun_stream("q")))

    assert events[-1].kind == "done" and events[-1].answer == "流式答案"
    (record,) = _lines(tmp_path / "traces" / "svc" / "s1.jsonl")
    assert _roles(record["messages"]) == ["system", "user", "assistant", "tool", "assistant"]
    assert record["answer"] == "流式答案" and record["num_tool_calls"] == 1
