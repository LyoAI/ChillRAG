"""Evaluator framework: score the answers file a batch run produced.

An :class:`Evaluator` turns one answers row into one :class:`Verdict`, and
each evaluator pairs with its own HTML page (:meth:`Evaluator.render_report`)
— what a scorer should show is its own business, so the page ships with the
scorer, not with the framework. Rows without a ``label`` (the reference
answer) are *skipped*, never failed: mixed batches stay scoreable, and the
skip is reported, not buried.

Aggregates are plain data (``label -> number``); rendering stays out of the
framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar


#: Strings a free-text answer or reference uses to say it found nothing.
_ABSTAIN = ("n/a", "na", "nan", "none", "null", "无", "未找到", "未提及",
            "不存在", "无法确定", "未知")


def abstains(text: str) -> bool:
    """Whether an answer or reference declares nothing found rather than a value.

    A reference that abstains flips the row's contract: the answer must
    abstain too, and any value it names counts as fabricated.
    """
    return not text.strip() or text.strip().lower() in _ABSTAIN


@dataclass
class Verdict:
    """One row's score, with the reason a human can read in the table.

    ``data`` carries whatever the evaluator's own page needs to show this row
    — parsed values, unit conversions, rationales. The framework never reads
    it; the evaluator's renderer does.
    """

    index: int                    # the answers row's index, joining back to its trace
    ok: bool
    score: float = 1.0            # 0..1; partial credit is an evaluator's choice
    detail: str = ""              # short machine tag, e.g. "exact", "round-tol"
    note: str = ""                # human explanation, e.g. "3.51亿 ≈ 35142万"
    data: dict[str, Any] = field(default_factory=dict)


class Evaluator(ABC):
    """Scores answers rows against their reference (``label``) values — and
    renders the page that shows the scoring.

    Subclass and override :meth:`score_row` plus :meth:`render_report`;
    override :meth:`aggregate` to add evaluator-specific statistics alongside
    the base accuracy.
    """

    #: Registry / CLI name.
    name: ClassVar[str]
    #: Answers-file columns: the model's answer, and the reference it must match.
    answer_field: ClassVar[str] = "answer"
    label_field: ClassVar[str] = "label"

    @abstractmethod
    def score_row(self, row: dict[str, Any]) -> Verdict:
        """Score one answers row. The row carries the model's answer, the
        ``label`` and any metadata the batch run passed through."""

    @abstractmethod
    def render_report(self, verdicts: list[Verdict], rows: dict[int, dict],
                      *, answers: str, skipped: int) -> str:
        """The evaluator's own page: one self-contained HTML document.

        ``rows`` maps each verdict's index back to its answers row, so the
        page can show whatever context the evaluator cares about. Return the
        whole document — the CLI writes it and opens a browser.
        """

    def aggregate(self, verdicts: list[Verdict]) -> dict[str, Any]:
        """Plain-number summary of a scored run; subclasses extend this."""
        n = len(verdicts)
        hits = sum(v.ok for v in verdicts)
        details: dict[str, int] = {}
        for v in verdicts:
            details[v.detail or "-"] = details.get(v.detail or "-", 0) + 1
        return {
            "scored": n,
            "accuracy": hits / n if n else 0.0,
            "correct": hits,
            "details": details,
        }


def evaluate(rows: list[dict[str, Any]], evaluator: Evaluator,
             ) -> tuple[list[Verdict], int]:
    """Score every row that names a reference; return verdicts and skip count.

    Verdicts keep input order, and each carries the row's own ``index`` when
    it has one, so results join back to the answers file and its traces.
    """
    verdicts: list[Verdict] = []
    skipped = 0
    for position, row in enumerate(rows):
        label = row.get(evaluator.label_field)
        if label is None or not str(label).strip():
            skipped += 1
            continue
        verdict = evaluator.score_row(row)
        try:
            verdict.index = int(row.get("index", position))
        except (TypeError, ValueError):
            verdict.index = position
        verdicts.append(verdict)
    return verdicts, skipped
