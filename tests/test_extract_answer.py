"""extract_answer: the caller's tag contract, unwrapped in place."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.extract_answer import main  # noqa: E402


def _answers(tmp_path, rows):
    path = tmp_path / "q.answers.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                    encoding="utf-8")
    return path


def test_wrapped_answers_are_extracted_in_place(tmp_path):
    path = _answers(tmp_path, [
        {"index": 0, "answer": "依据如下……<answer>92,599,365.5万元</answer>", "label": "92599365.5万元"},
        {"index": 1, "answer": "<answer>1.87%</answer>", "label": "1.87%"},
    ])
    assert main([str(path)]) == 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["answer"] == "92,599,365.5万元"
    assert rows[1]["answer"] == "1.87%"


def test_untagged_rows_survive_and_the_run_is_idempotent(tmp_path):
    path = _answers(tmp_path, [
        {"index": 0, "answer": "旧格式回答 42", "label": "42"},
        {"index": 1, "answer": "<answer>42</answer>", "label": "42"},
    ])
    assert main([str(path)]) == 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["answer"] == "旧格式回答 42", "no tag, no touch"

    assert main([str(path)]) == 0, "a second run finds nothing to do and changes nothing"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[1]["answer"] == "42"
