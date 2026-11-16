"""The eval CLI end to end: score a real answers file, print and export."""

import json

from chillrag.eval.__main__ import main


def _answers(tmp_path, rows):
    path = tmp_path / "q.answers.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                    encoding="utf-8")
    return path


def test_a_mixed_file_scores_and_exports(tmp_path):
    path = _answers(tmp_path, [
        {"index": 0, "question": "资产总额", "answer": "92,599,365.5万元", "label": "92599365.5万元"},
        {"index": 1, "question": "不良率", "answer": "1.87%", "label": "1.87%"},
        {"index": 2, "question": "存款总额", "answer": "4,000万元", "label": "3,900万元"},
        {"index": 3, "question": "无标准答案", "answer": "随便什么"},
    ])
    out = tmp_path / "verdicts.jsonl"
    assert main([str(path), "--out", str(out)]) == 0

    verdicts = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [v["index"] for v in verdicts] == [0, 1, 2], "the unlabeled row is skipped"
    assert [v["ok"] for v in verdicts] == [True, True, False]
    assert verdicts[1]["detail"] == "exact"
    assert verdicts[2]["detail"] == "mismatch"


def test_no_labels_at_all_is_a_clean_exit(tmp_path):
    path = _answers(tmp_path, [{"index": 0, "question": "q", "answer": "a"}])
    assert main([str(path)]) == 0


def test_the_exact_evaluator_is_a_cli_choice(tmp_path):
    path = _answers(tmp_path, [{"index": 0, "question": "q", "answer": "B.", "label": "b"}])
    assert main([str(path), "--evaluator", "exact"]) == 0
