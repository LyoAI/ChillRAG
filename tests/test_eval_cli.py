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
    embedded = report.read_text(encoding="utf-8").split('id="eval-data">')[1]
    assert "</script><b>injected</b>" not in embedded.split("</script>")[0]


def test_no_report_skips_the_artifact(tmp_path):
    path = _answers(tmp_path, [{"index": 0, "question": "q", "answer": "a", "label": "a"}])
    assert main([str(path), "--no-report", "--no-open"]) == 0
    assert list(tmp_path.iterdir()) == [path]
