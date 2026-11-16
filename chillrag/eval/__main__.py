"""CLI: score a batch answers file against its ``label`` column.

    python -m chillrag.eval runs/q.answers.jsonl
    python -m chillrag.eval runs/q.xlsx --evaluator exact --out verdicts.jsonl

Each run also renders a self-contained HTML report next to the answers file
(``q.answers.eval.html``) — stat tiles, the wrong answers up front, one click
expands a row — and opens it in the browser (``--no-open`` keeps it shut,
``--no-report`` skips it). Rows without a ``label`` are skipped and reported,
never failed. Exit code is 0 whenever scoring ran; accuracy itself is the
output, not a verdict.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

from rich.console import Console
from rich.table import Table

from chillrag.eval import EVALUATORS, evaluate
from chillrag.eval.report import render_eval_html
from chillrag.utils.answers import read_answers

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("answers", type=Path, help="the answers file from a batch run")
    parser.add_argument("--evaluator", choices=sorted(EVALUATORS), default="fin-numeric",
                        help="which scorer to apply (default: fin-numeric)")
    parser.add_argument("--answer-field", help="the model's answer column (default: %(default)s)",
                        default="answer")
    parser.add_argument("--label-field", help="reference column (default: %(default)s)",
                        default="label")
    parser.add_argument("--unit-field", help="reference-unit column, numeric scorer only",
                        default="unit")
    parser.add_argument("--out", type=Path, help="also write per-row verdicts as JSONL")
    parser.add_argument("--report", type=Path,
                        help="HTML report path (default: alongside the answers file)")
    parser.add_argument("--no-report", action="store_true", help="skip the HTML report")
    parser.add_argument("--no-open", action="store_true",
                        help="write the report, don't open a browser")
    parser.add_argument("--max-rows", type=int, default=40,
                        help="failing rows to print in full (default 40)")
    args = parser.parse_args(argv)

    evaluator = EVALUATORS[args.evaluator]()
    evaluator.answer_field = args.answer_field
    evaluator.label_field = args.label_field
    if hasattr(evaluator, "unit_field"):
        evaluator.unit_field = args.unit_field

    rows = read_answers(args.answers)
    verdicts, skipped = evaluate(rows, evaluator)
    if not verdicts:
        print(f"nothing scored in {args.answers} — no row carries a "
              f"{args.label_field!r} column value")
        return 0
    # Answers stream in completion order; join verdicts to rows by index.
    by_index: dict[int, dict] = {}
    for position, row in enumerate(rows):
        try:
            by_index[int(row.get("index", position))] = row
        except (TypeError, ValueError):
            by_index[position] = row

    stats = evaluator.aggregate(verdicts)
    stats["name"] = evaluator.name
    summary = Table.grid(pad_edge=False)
    summary.add_column(style="dim", width=18)
    summary.add_column()
    summary.add_row("evaluator", evaluator.name)
    summary.add_row("scored", f"{stats['scored']} · skipped {skipped} (no {args.label_field!r})")
    summary.add_row("accuracy", f"{stats['accuracy']:.1%} ({stats['correct']}/{stats['scored']})")
    for key, value in stats.items():
        if key not in ("scored", "accuracy", "correct", "details", "name"):
            summary.add_row(key, f"{value:.1%}" if isinstance(value, float) else str(value))
    summary.add_row("details", " · ".join(f"{k} {n}" for k, n in stats["details"].items()))
    console.print(summary)

    misses = [v for v in verdicts if not v.ok]
    if misses:
        table = Table(title=f"not matching ({len(misses)})"
                      if len(misses) <= args.max_rows else
                      f"not matching ({len(misses)}, first {args.max_rows})")
        for column, just in (("index", "right"), ("detail", "left")):
            table.add_column(column, justify=just)
        table.add_column("question", max_width=36, overflow="ellipsis")
        table.add_column(args.label_field, max_width=20, overflow="ellipsis")
        table.add_column("answer", max_width=36, overflow="ellipsis")
        table.add_column("note", max_width=30, overflow="ellipsis")
        for v in misses[: args.max_rows]:
            row = by_index.get(v.index, {})
            table.add_row(str(v.index), v.detail, str(row.get("question", "")),
                          str(row.get(args.label_field, "")),
                          str(row.get(args.answer_field, "")), v.note)
        console.print(table)

    if args.out:
        args.out.write_text(
            "\n".join(json.dumps({"index": v.index, "ok": v.ok, "score": v.score,
                                  "detail": v.detail, "note": v.note}, ensure_ascii=False)
                      for v in verdicts) + "\n", encoding="utf-8")
        print(f"verdicts → {args.out}")

    if not args.no_report:
        payload = {
            "evaluator": evaluator.name,
            "answers": str(args.answers),
            "skipped": skipped,
            "stats": stats,
            "verdicts": [{"index": v.index, "ok": v.ok, "detail": v.detail, "note": v.note,
                          "question": str(by_index.get(v.index, {}).get("question", "")),
                          "label": str(by_index.get(v.index, {}).get(args.label_field, "")),
                          "answer": str(by_index.get(v.index, {}).get(args.answer_field, ""))}
                         for v in verdicts],
        }
        report_path = args.report or args.answers.with_name(args.answers.stem + ".eval.html")
        report_path.write_text(render_eval_html(payload), encoding="utf-8")
        print(f"report → {report_path}")
        if not args.no_open:
            webbrowser.open(f"file://{report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
