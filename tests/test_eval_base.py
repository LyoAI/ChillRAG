"""Evaluator framework: the ``score_row`` contract, skipping, aggregation."""

import pytest

from chillrag.eval import Evaluator, Verdict, evaluate


class _Always(Evaluator):
    name = "always"

    def score_row(self, row: dict) -> Verdict:
        return Verdict(0, True, 1.0, "exact")


def test_the_framework_itself_cannot_score():
    with pytest.raises(TypeError):
        Evaluator()  # abstract — a scorer is a subclass's job


def test_rows_without_a_label_are_skipped_not_failed():
    rows = [
        {"question": "q1", "label": "x"},
        {"question": "q2"},
        {"question": "q3", "label": "  "},
    ]
    verdicts, skipped = evaluate(rows, _Always())
    assert skipped == 2
    assert [v.ok for v in verdicts] == [True]
    assert verdicts[0].index == 0


def test_the_verdict_index_follows_the_row():
    rows = [
        {"question": "q9", "index": 9, "label": "x"},
        {"question": "q2", "label": "y", "index": "not-a-number"},
    ]
    verdicts, _ = evaluate(rows, _Always())
    assert [v.index for v in verdicts] == [9, 1], "unreadable index falls back to position"


def test_default_aggregate_reports_accuracy_and_detail_counts():
    verdicts = [Verdict(0, True, 1.0, "exact"), Verdict(1, False, 0.0, "mismatch"),
                Verdict(2, True, 1.0, "round-tol")]
    stats = _Always().aggregate(verdicts)
    assert stats["scored"] == 3 and stats["correct"] == 2
    assert stats["accuracy"] == pytest.approx(2 / 3)
    assert stats["details"] == {"exact": 1, "mismatch": 1, "round-tol": 1}


def test_an_empty_run_aggregates_to_zero_not_a_crash():
    stats = _Always().aggregate([])
    assert stats["scored"] == 0 and stats["accuracy"] == 0.0
