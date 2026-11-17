"""The eval CLI end to end: score a real answers file, print, export, report."""

import json

import pytest

from chillrag.eval.__main__ import main


@pytest.fixture(autouse=True)
def no_browser(monkeypatch):
    opened = []
    monkeypatch.setattr("chillrag.eval.__main__.webbrowser.open",
                        lambda url: opened.append(url) or True)
    return opened


def _answers(tmp_path, rows, name="q.answers.jsonl"):
    path = tmp_path / name
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


def test_answer_and_label_fields_can_be_renamed(tmp_path):
    path = _answers(tmp_path, [
        {"index": 0, "question": "q", "output": "3.51亿元", "answer_key": "35142万元"},
    ])
    out = tmp_path / "v.jsonl"
    assert main([str(path), "--answer-field", "output", "--label-field", "answer_key",
                 "--out", str(out), "--no-report", "--no-open"]) == 0
    verdicts = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert verdicts[0]["ok"] and verdicts[0]["detail"] == "exact"


def test_the_report_lands_next_to_the_answers_and_opens(tmp_path, no_browser):
    path = _answers(tmp_path, [
        {"index": 0, "question": "资产总额", "answer": "92,599,365.5万元", "label": "92599365.5万元"},
        {"index": 1, "question": "营业成本", "answer": "3.9亿", "label": "35142万元"},
    ])
    report = tmp_path / "q.answers.eval.html"
    assert main([str(path), "--no-open"]) == 0
    assert report.exists()
    html = report.read_text(encoding="utf-8")
    assert "50.0%" in html and "3.9亿" in html and "资产总额" in html
    assert {p.name for p in tmp_path.iterdir()} == {
        "q.answers.jsonl", "q.answers.eval.html"}
    # --no-open keeps the browser shut; the default opens the report.
    assert no_browser == []
    assert main([str(path)]) == 0
    assert no_browser and no_browser[0].startswith("file://")


def test_hostile_content_cannot_break_the_report(tmp_path):
    path = _answers(tmp_path, [
        {"index": 0, "question": "</script><b>injected</b>", "answer": "x", "label": "y"},
    ])
    report = tmp_path / "q.answers.eval.html"
    assert main([str(path), "--no-open"]) == 0
    embedded = report.read_text(encoding="utf-8").split('id="page-data">')[1]
    assert "</script><b>injected</b>" not in embedded.split("</script>")[0]


def test_the_report_carries_the_evaluators_own_marks(tmp_path):
    """The numeric page marks every question right or wrong, by its rule."""
    path = _answers(tmp_path, [
        {"index": 0, "question": "q1", "answer": "3.51亿元", "label": "35142万元"},
        {"index": 1, "question": "q2", "answer": "35142.1万元", "label": "35142.05万元"},
        {"index": 2, "question": "q3", "answer": "35100万元", "label": "35142万元"},
    ])
    report = tmp_path / "q.answers.eval.html"
    assert main([str(path), "--no-open"]) == 0
    html = report.read_text(encoding="utf-8")
    assert "66.7%" in html and "ChillRAG Eval · fin-numeric" in html
    for rule in ("exact", "round-tol", "mismatch"):
        assert rule in html, f"each row shows the rule that decided: {rule}"


def test_no_report_skips_the_artifact(tmp_path):
    path = _answers(tmp_path, [{"index": 0, "question": "q", "answer": "a", "label": "a"}])
    assert main([str(path), "--no-report", "--no-open"]) == 0
    assert list(tmp_path.iterdir()) == [path]
