"""retry_failed: the failed rows of an answers file become a clean retry
input — output fields stripped, input order kept, failures classified."""

import json
from pathlib import Path

import pytest

from chillrag.utils import retry_failed


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


def test_error_rows_become_clean_input(tmp_path):
    path = _answers(tmp_path, [
        _row(0, "q1"),
        _row(2, "q3", "error", error="ConnectionError: reset by peer", attempts=3),
        _row(1, "q2", "error", error="TimeoutError: answering exceeded 600s"),
    ])
    out = tmp_path / "q.retry.jsonl"
    assert retry_failed.main([str(path)]) == 0

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["question"] for r in rows] == ["q2", "q3"], "input order, not completion order"
    assert rows[0]["query"] == "q2" and rows[0]["folder"] == "bank"
    joined = out.read_text(encoding="utf-8")
    for dropped in ("status", "answer", "trace_id", "error", "duration_s", "attempts", "index"):
        assert f'"{dropped}"' not in joined, f"the output-only field {dropped!r} must not ride along"


def test_empty_rows_need_the_flag(tmp_path):
    rows = [_row(0, "q1", "empty"), _row(1, "q2")]
    path = _answers(tmp_path, rows)
    out = tmp_path / "q.retry.jsonl"
    assert retry_failed.main([str(path)]) == 0
    assert not out.exists(), "empty rows stay out unless asked for"

    assert retry_failed.main([str(path), "--include-empty", "--out", str(out)]) == 0
    assert [json.loads(line)["question"] for line in out.read_text().splitlines()] == ["q1"]


def test_nothing_to_retry_is_a_clean_exit(tmp_path):
    path = _answers(tmp_path, [_row(0, "q1")])
    assert retry_failed.main([str(path)]) == 0


def test_an_xlsx_answers_file_is_refused(tmp_path):
    path = tmp_path / "q.answers.xlsx"
    path.write_bytes(b"not really")
    with pytest.raises(SystemExit, match="jsonl"):
        retry_failed.main([str(path)])


def test_classify_names_the_remedy():
    assert "CHILLRAG_ANSWER_TIMEOUT_S" in retry_failed.classify(
        "TimeoutError: answering exceeded 600s")
    assert "transient" in retry_failed.classify("APIConnectionError: reset by peer")
    assert "inspect" in retry_failed.classify("ValueError: no corpus")
