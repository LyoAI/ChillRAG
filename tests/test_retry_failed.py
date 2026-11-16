"""retry_failed: the failed rows of an answers file, re-answered and patched
back into the same file — no new artifacts, healed rows look first-run.

The tests drive ``main`` over real temp answers files and a fake tool, and
assert what lands on disk.
"""

import json
from pathlib import Path

import pytest

from chillrag import batch_request
from chillrag.utils import retry_failed
from test_batch import FakeTool


@pytest.fixture
def fake_tool(monkeypatch):
    tool = FakeTool()
    monkeypatch.setattr(batch_request, "_RETRY_DELAYS", (0,))
    monkeypatch.setattr(batch_request, "_load_tool", lambda folders: tool)
    return tool


def _row(index: int, question: str, status: str = "ok", **extra) -> dict:
    row = {"index": index, "question": question, "folder": "bank", "user": "batch",
           "query": question, "answer": f"ans:{question}" if status == "ok" else "",
           "status": status, "duration_s": 1.0}
    row.update(extra)
    return row


def _answers(tmp_path, rows) -> Path:
    path = tmp_path / "q.answers.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                    encoding="utf-8")
    return path


def test_error_rows_are_reanswered_in_place(fake_tool, tmp_path):
    path = _answers(tmp_path, [
        _row(0, "q1"),
        _row(1, "q2", "error", error="ConnectionError: reset by peer", attempts=3),
    ])
    assert retry_failed.main([str(path)]) == 0

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["status"] for r in rows] == ["ok", "ok"]
    assert rows[1]["answer"] == "ans:q2" and rows[1]["trace_id"]
    assert rows[1]["query"] == "q2", "the row's own metadata survives"
    for gone in ("error", "attempts"):
        assert gone not in rows[1], "a healed row reads as first-run"
    assert rows[0] == _row(0, "q1"), "answered rows are untouched"
    assert {p.name for p in tmp_path.iterdir()} == {"q.answers.jsonl", "traces"}, "no new files"


def test_empty_rows_wait_for_the_flag(fake_tool, tmp_path):
    path = _answers(tmp_path, [_row(0, "q1", "empty")])
    assert retry_failed.main([str(path)]) == 0
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "empty"

    assert retry_failed.main([str(path), "--include-empty"]) == 0
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["status"] == "ok" and row["answer"] == "ans:q1"


def test_rows_that_fail_again_stay_error(fake_tool, tmp_path):
    path = _answers(tmp_path, [_row(0, "boom", "error", error="ConnectionError: x")])
    assert retry_failed.main([str(path)]) == 1
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["status"] == "error" and "RuntimeError" in row["error"]
    assert "trace_id" not in row


def test_an_xlsx_answers_file_is_patched_in_place(fake_tool, tmp_path):
    import openpyxl

    path = tmp_path / "q.xlsx"
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["index", "question", "folder", "user", "answer", "status", "duration_s"])
    sheet.append([0, "q1", "bank", "batch", "ans:q1", "ok", 1.0])
    sheet.append([1, "q2", "bank", "batch", "", "error", 2.0])
    book.save(path)

    assert retry_failed.main([str(path)]) == 0
    rows = list(openpyxl.load_workbook(path).active.iter_rows(values_only=True))
    header, data = rows[0], rows[1:]
    healed = dict(zip(header, data[1]))
    assert healed["status"] == "ok" and healed["answer"] == "ans:q2"
    assert dict(zip(header, data[0]))["answer"] == "ans:q1"
    assert {p.name for p in tmp_path.iterdir()} == {"q.xlsx", "traces"}, "no new files"


def test_nothing_to_retry_leaves_the_file_alone(fake_tool, tmp_path):
    path = _answers(tmp_path, [_row(0, "q1")])
    before = path.read_text(encoding="utf-8")
    assert retry_failed.main([str(path)]) == 0
    assert path.read_text(encoding="utf-8") == before


def test_other_suffixes_are_refused(tmp_path):
    path = tmp_path / "q.answers.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="jsonl or .xlsx"):
        retry_failed.main([str(path)])


def test_classify_names_the_remedy():
    assert "CHILLRAG_ANSWER_TIMEOUT_S" in retry_failed.classify(
        "TimeoutError: answering exceeded 600s")
    assert "transient" in retry_failed.classify("APIConnectionError: reset by peer")
    assert "inspect" in retry_failed.classify("ValueError: no corpus")
