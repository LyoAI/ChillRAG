"""HTML trace report: episodes embedded as data, deep-linkable, browser-off optional."""

import json
from pathlib import Path

import pytest

from chillrag.utils import trace_view


def _session_dir(tmp_path: Path) -> Path:
    d = tmp_path / "traces" / "svc"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record(question: str, answer: str, trace_id: str) -> dict:
    return {
        "trace_id": trace_id, "ts": "2026-11-10T03:01:29+00:00",
        "user": "svc", "session_id": "s1", "source": "batch", "folder": "bank",
        "mode": "agent", "lang": "zh",
        "question": question, "answer": answer,
        "num_tool_calls": 1, "terminated": "answer", "total_tokens": 42,
        "duration_s": 1.5,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "bm25_search", "arguments": "{\"query\": \"x\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "[\"hit\"]"},
            {"role": "assistant", "content": answer},
        ],
    }


@pytest.fixture
def session_file(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_view, "TRACE_DIR", tmp_path / "traces")
    path = _session_dir(tmp_path) / "s1.jsonl"
    path.write_text(json.dumps(_record("q1", "a1", "abc123def456")) + "\n" +
                    json.dumps(_record("q2", "a2", "def456abc123")) + "\n", encoding="utf-8")
    return path


def _embedded(html: str) -> dict:
    return json.loads(html.split('id="cr-data">')[1].split("</script>")[0])


def test_the_report_embeds_every_episode_and_its_transcript(session_file):
    html = trace_view.render_traces_html(
        trace_view._build_payload(trace_view._session_files(None)))
    data = _embedded(html)
    assert data["trace_dir"].endswith("traces")
    (session,) = data["sessions"]
    assert session["n"] == 2 and session["user"] == "svc" and session["session"] == "s1"
    assert [e["question"] for e in session["episodes"]] == ["q1", "q2"]
    episode = session["episodes"][0]
    assert episode["tool_chain"] == ["bm25_search"]
    assert [m["role"] for m in episode["messages"]] == [
        "user", "assistant", "tool", "assistant"]


def test_script_closing_tags_in_content_cannot_break_the_document(session_file, tmp_path):
    path = _session_dir(tmp_path) / "sneaky.jsonl"
    record = _record("</script><b>injected</b>", "a", "1234567890ab")
    path.write_text(json.dumps(record), encoding="utf-8")
    html = trace_view.render_traces_html(
        trace_view._build_payload(trace_view._session_files(None)))
    assert "</script><b>injected</b>" not in html.split('id="cr-data">')[1].split("</script>")[0]


def test_main_writes_the_report_and_opens_it(tmp_path, monkeypatch, session_file):
    opened = []
    monkeypatch.setattr(trace_view.webbrowser, "open", lambda url: opened.append(url) or True)
    out = tmp_path / "report.html"
    code = trace_view.main(["--out", str(out)])
    assert code == 0 and out.exists()
    assert opened == [f"file://{out.resolve()}"]


def test_a_trace_argument_deep_links_the_episode(tmp_path, monkeypatch, session_file):
    opened = []
    monkeypatch.setattr(trace_view.webbrowser, "open", lambda url: opened.append(url) or True)
    code = trace_view.main(["--out", str(tmp_path / "r.html"), "s1", "abc123de"])
    assert code == 0
    assert opened[0].endswith("#trace-abc123de")


def test_an_unknown_session_is_a_clean_exit(session_file):
    assert trace_view.main(["nope"]) == 1
