"""Batch driver: rows in, agent episodes out, traces bound to each row.

The tests drive ``main`` end to end over a fake tool and real temp files —
what is asserted is the answers file, the concurrency ceiling, the failure
isolation, and the trace records the episodes write.
"""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from chillrag import batch_request
from chillrag.tracing import record_episode


class FakeTool:
    """Slow, observable stand-in for the agent: counts concurrent episodes.

    Like the real agent, an episode records itself on success and stays
    silent on failure.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls: list[tuple[str, tuple[int, ...]]] = []
        self.fail_first: dict[str, int] = {}  # question -> how many calls raise first

    def answer_by_question(self, question, source_id_list=(), **kwargs):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append((question, tuple(source_id_list)))
        try:
            time.sleep(0.05)
            if question.startswith("boom"):
                raise RuntimeError("the model is unhappy")
            if question.startswith("slow"):
                raise TimeoutError("answering exceeded 600s "
                                   "(raise ANSWER_TIMEOUT_S, or lower CHILLRAG_MAX_TOOL_CALLS)")
            with self._lock:
                flaky = self.fail_first.get(question, 0)
            if flaky:
                with self._lock:
                    self.fail_first[question] = flaky - 1
                raise ConnectionError("network hiccup")
            answer = f"ans:{question}"
        finally:
            with self._lock:
                self.active -= 1
        record_episode(SimpleNamespace(
            question=question, answer=answer, num_tool_calls=1, terminated="answer",
            total_tokens=10, duration_s=0.01,
            messages=[{"role": "user", "content": question}]))
        return answer


@pytest.fixture
def fake_tool(monkeypatch):
    tool = FakeTool()
    monkeypatch.setattr(batch_request, "_load_tool", lambda folders: tool)
    return tool


def _jsonl(path: Path, rows: list) -> Path:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


def _answers(out: Path) -> list[dict]:
    """The answers file streams in completion order; index restores input order."""
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    return sorted(records, key=lambda r: r["index"])


# ------------------------------------------------------------------ happy --

def test_a_batch_answers_every_row_in_order(fake_tool, tmp_path):
    src = _jsonl(tmp_path / "q.jsonl", [
        {"question": "q1", "user": "kainan", "note": "hello"},
        {"question": "q2"},
    ])
    out = tmp_path / "out.jsonl"
    code = batch_request.main([str(src), "--folder", "bank", "--out", str(out)])
    assert code == 0
    records = _answers(out)
    assert [r["question"] for r in records] == ["q1", "q2"]
    assert [r["answer"] for r in records] == ["ans:q1", "ans:q2"]
    assert all(r["status"] == "ok" and r["folder"] == "bank" for r in records)
    assert records[0]["note"] == "hello", "unknown columns ride along to the output"


def test_every_query_traces_in_its_own_session(fake_tool, tmp_path):
    src = _jsonl(tmp_path / "q.jsonl", [{"question": "q1"}, {"question": "q2"}])
    out = tmp_path / "out.jsonl"
    code = batch_request.main([str(src), "--folder", "bank", "--out", str(out)])
    assert code == 0

    # One session per query: two rows, two files, one episode each, each
    # named batch-<session_id> with an id of its own.
    files = sorted((tmp_path / "traces" / "batch").glob("*.jsonl"))
    assert len(files) == 2, f"expected one session per query, found {files}"
    records = [json.loads(p.read_text(encoding="utf-8").strip()) for p in files]
    assert sorted(r["question"] for r in records) == ["q1", "q2"]
    assert all(r["source"] == "batch" and r["user"] == "batch" for r in records)
    sessions = {r["session_id"] for r in records}
    assert len(sessions) == 2
    assert all(s.startswith("batch-") and len(s) == len("batch-") + 12 for s in sessions)
    assert {r["session_id"] for r in _answers(out)} == sessions


def test_a_rows_session_id_names_its_own_file(fake_tool, tmp_path):
    """A session is a multi-turn conversation; a batch row is not one. A row's
    session_id names that single episode's file — there is no batch-wide flag
    that would merge rows into a shared conversation."""
    src = _jsonl(tmp_path / "q.jsonl", [{"question": "q1", "session_id": "pinned"}])
    batch_request.main([str(src), "--folder", "bank", "--out", str(tmp_path / "out.jsonl")])

    (path,) = (tmp_path / "traces" / "batch").glob("*.jsonl")
    assert path.name == "pinned.jsonl"
    assert json.loads(path.read_text(encoding="utf-8"))["session_id"] == "pinned"


def test_each_answer_row_names_the_trace_that_produced_it(fake_tool, tmp_path):
    """A result joins back to its transcript by trace_id, not by guessing
    which line in which session file matches the question."""
    src = _jsonl(tmp_path / "q.jsonl", [{"question": "q1"}, {"question": "q2"}])
    out = tmp_path / "out.jsonl"
    batch_request.main([str(src), "--folder", "bank", "--out", str(out)])

    trace_ids = set()
    for path in (tmp_path / "traces").glob("*/*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            trace_ids.add(json.loads(line)["trace_id"])
    answers = _answers(out)
    assert len(trace_ids) == 2
    assert {r["trace_id"] for r in answers} == trace_ids


def test_row_identity_overrides_the_run_defaults(fake_tool, tmp_path):
    src = _jsonl(tmp_path / "q.jsonl", [
        {"question": "q1", "user": "alice", "session_id": "s-a"},
        {"question": "q2", "session_id": "s-b"},
    ])
    batch_request.main([str(src), "--folder", "bank", "--out", str(tmp_path / "out.jsonl")])

    users = {p.parent.name for p in (tmp_path / "traces").glob("*/*.jsonl")}
    assert users == {"alice", "batch"}
    assert (tmp_path / "traces" / "alice" / "s-a.jsonl").exists()
    assert (tmp_path / "traces" / "batch" / "s-b.jsonl").exists()


def test_each_row_retrieves_from_its_own_folder(fake_tool, tmp_path):
    """The handle carries every corpus in the batch, so a row's folder must be
    named per call — an unnamed request lands on whichever corpus was attached
    last, and a mixed-folder batch would silently answer from one company."""
    from chillrag.rag_tool import source_id_for

    src = _jsonl(tmp_path / "q.jsonl", [
        {"question": "q1", "folder": "byd"},
        {"question": "q2", "folder": "cqb"},
        {"question": "q3"},
    ])
    out = tmp_path / "out.jsonl"
    code = batch_request.main([str(src), "--folder", "byd", "--out", str(out)])
    assert code == 0
    scope = {q: ids for q, ids in fake_tool.calls}
    assert scope["q1"] == (source_id_for("byd"),)
    assert scope["q2"] == (source_id_for("cqb"),)
    assert scope["q3"] == (source_id_for("byd"),), "a row falls back to --folder"


def test_user_and_source_flags_fill_unnamed_rows(fake_tool, tmp_path):
    src = _jsonl(tmp_path / "q.jsonl", [{"question": "q1"}])
    batch_request.main([str(src), "--folder", "bank", "--user", "kainan",
                        "--source", "eval", "--out", str(tmp_path / "out.jsonl")])

    (path,) = (tmp_path / "traces" / "kainan").glob("*.jsonl")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["user"] == "kainan" and record["source"] == "eval"


def test_bare_jsonl_strings_are_questions(fake_tool, tmp_path):
    src = _jsonl(tmp_path / "q.jsonl", ["solo question"])
    out = tmp_path / "out.jsonl"
    batch_request.main([str(src), "--folder", "bank", "--out", str(out)])
    assert _answers(out)[0]["answer"] == "ans:solo question"


# ------------------------------------------------------------ concurrency --

def test_concurrency_never_exceeds_the_cap(fake_tool, tmp_path):
    src = _jsonl(tmp_path / "q.jsonl", [{"question": f"q{n}"} for n in range(6)])
    code = batch_request.main([str(src), "--folder", "bank", "-c", "2",
                               "--out", str(tmp_path / "out.jsonl")])
    assert code == 0
    assert fake_tool.max_active <= 2
    assert fake_tool.max_active > 1, "the queue must actually overlap episodes"


# ---------------------------------------------------------------- failure --

def test_a_flaky_row_is_retried_until_it_answers(fake_tool, tmp_path, monkeypatch):
    """A transient failure is invisible in the final output except for the
    attempts count: the row answers, and its trace_id names the attempt that
    worked, since a failed attempt never recorded an episode."""
    monkeypatch.setattr(batch_request, "_RETRY_DELAYS", (0,))
    fake_tool.fail_first = {"flaky": 1}
    src = _jsonl(tmp_path / "q.jsonl", [{"question": "flaky"}, {"question": "q2"}])
    out = tmp_path / "out.jsonl"
    code = batch_request.main([str(src), "--folder", "bank", "--retries", "2",
                               "--out", str(out)])
    assert code == 0
    records = _answers(out)
    assert [r["status"] for r in records] == ["ok", "ok"]
    assert records[0]["answer"] == "ans:flaky"
    assert records[0]["attempts"] == 2 and "attempts" not in records[1]
    assert records[0]["trace_id"], "the successful attempt names its trace"


def test_exhausted_retries_end_in_the_error_row(fake_tool, tmp_path, monkeypatch):
    monkeypatch.setattr(batch_request, "_RETRY_DELAYS", (0,))
    src = _jsonl(tmp_path / "q.jsonl", [{"question": "boom"}])
    out = tmp_path / "out.jsonl"
    code = batch_request.main([str(src), "--folder", "bank", "--retries", "2",
                               "--out", str(out)])
    assert code == 1
    (record,) = _answers(out)
    assert record["status"] == "error" and record["attempts"] == 3
    assert "RuntimeError" in record["error"]
    assert "trace_id" not in record


def test_a_timeout_is_not_retried(fake_tool, tmp_path, monkeypatch):
    """A retry would burn a full ANSWER_TIMEOUT_S window to learn the same
    lesson; the row fails at once and the message names the knob to raise."""
    monkeypatch.setattr(batch_request, "_RETRY_DELAYS", (0,))
    src = _jsonl(tmp_path / "q.jsonl", [{"question": "slow question"}])
    out = tmp_path / "out.jsonl"
    code = batch_request.main([str(src), "--folder", "bank", "--retries", "2",
                               "--out", str(out)])
    assert code == 1
    (record,) = _answers(out)
    assert record["status"] == "error"
    assert "attempts" not in record, "a timeout got no second attempt"
    assert "ANSWER_TIMEOUT_S" in record["error"]
    assert "trace_id" not in record


def test_one_bad_row_does_not_sink_the_batch(fake_tool, tmp_path, monkeypatch):
    monkeypatch.setattr(batch_request, "_RETRY_DELAYS", (0,))
    src = _jsonl(tmp_path / "q.jsonl", [{"question": "q1"}, {"question": "boom"},
                                        {"question": "q3"}])
    out = tmp_path / "out.jsonl"
    code = batch_request.main([str(src), "--folder", "bank", "--out", str(out)])
    assert code == 1
    records = _answers(out)
    assert [r["status"] for r in records] == ["ok", "error", "ok"]
    assert "RuntimeError" in records[1]["error"]
    assert "trace_id" not in records[1], "a failed row recorded no episode"


def test_a_row_without_a_question_is_rejected_with_its_number(fake_tool, tmp_path):
    src = _jsonl(tmp_path / "q.jsonl", [{"question": "q1"}, {"note": "no question"}])
    with pytest.raises(SystemExit, match="row 2"):
        batch_request.main([str(src), "--folder", "bank", "--out", str(tmp_path / "o.jsonl")])


def test_a_row_without_any_folder_names_the_prepared(fake_tool, tmp_path, monkeypatch):
    src = _jsonl(tmp_path / "q.jsonl", [{"question": "q1"}])
    monkeypatch.setattr(batch_request, "_prepared_folders", lambda: ["bank-a", "bank-b"])
    with pytest.raises(SystemExit, match="bank-a, bank-b"):
        batch_request.main([str(src), "--out", str(tmp_path / "o.jsonl")])


# ------------------------------------------------------------------ excel --

def test_an_excel_sheet_round_trips(fake_tool, tmp_path):
    import openpyxl

    src = tmp_path / "q.xlsx"
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["query", "user", "when"])
    sheet.append(["from excel", "kainan", "2026-11-10"])
    sheet.append([42])  # a numeric question is still a question
    book.save(src)

    out = tmp_path / "out.xlsx"
    code = batch_request.main([str(src), "--folder", "bank", "--question-field", "query",
                               "--out", str(out)])
    assert code == 0
    book = openpyxl.load_workbook(out)
    rows = list(book.active.iter_rows(values_only=True))
    header, data = rows[0], rows[1:]
    # The question column is normalized to "question"; the input's "query"
    # name and the passthrough "when" column survive.
    assert {"user", "when", "question", "answer", "status"} <= set(header)
    assert [dict(zip(header, r))["answer"] for r in data] == [
        "ans:from excel", "ans:42"]
    assert dict(zip(header, data[0]))["when"] == "2026-11-10"
    assert dict(zip(header, data[0]))["status"] == "ok"
