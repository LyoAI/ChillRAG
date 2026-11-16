"""Retry the unanswered rows of a batch answers file, in place.

Reads ``<input>.answers.jsonl`` (or ``.xlsx``), re-answers the rows that did
not answer — ``status: "error"`` or ``"empty"`` — through the same engine as
``batch_request``, and patches the results back into the same file. No new
artifacts: the answers stay one file in input order, and each patched row
keeps its identity (folder, user, session) so its trace joins exactly as
before. The rewrite is atomic — answers land first, then one replace — so a
crash mid-run never corrupts what was there.

    python -m chillrag.utils.retry_failed runs/q.answers.jsonl
    python -m chillrag.utils.retry_failed runs/q.xlsx -c 4

Timeouts still never retry (``--retries`` covers transient network errors
only); rows that fail again stay ``status: "error"`` and the closing
breakdown names the remedy — usually raising ``CHILLRAG_MAX_TOOL_CALLS`` /
``CHILLRAG_ANSWER_TIMEOUT_S``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path

from chillrag import batch_request

#: Fields a retry attempt overwrites; the rest of the record is the row's own
#: identity and rides along untouched.
OUTPUT_FIELDS = frozenset(
    {"index", "answer", "status", "duration_s", "trace_id", "error", "attempts"})


def _read_records(path: Path) -> list[dict]:
    """Every answers record, in file order — jsonl lines or xlsx data rows."""
    if path.suffix.lower() == ".xlsx":
        import openpyxl

        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        rows = list(book.active.iter_rows(values_only=True))
        if not rows:
            return []
        header = [str(cell).strip() if cell is not None else "" for cell in rows[0]]
        return [dict(zip(header, row)) for row in rows[1:]
                if any(cell is not None for cell in row)]
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise SystemExit(f"expected a .jsonl or .xlsx answers file, got {path.name!r}")


def _failed(records: list[dict]) -> list[tuple[dict, dict]]:
    """``(record, clean_input_row)`` for the unanswered rows, input order.

    Empty counts as unanswered: an empty answer is not evidence the corpus
    lacks the fact — the agent may have been forced off its search early.
    """
    bad = [r for r in records if r.get("status") in ("error", "empty")]
    bad.sort(key=lambda r: r.get("index", 0))
    return [(r, {k: v for k, v in r.items() if k not in OUTPUT_FIELDS}) for r in bad]


def _patch(record: dict, outcome) -> None:
    """Overwrite a record with its retry outcome; healed rows look first-run."""
    record["answer"] = outcome.answer
    record["status"] = outcome.status
    record["duration_s"] = round(outcome.duration_s, 3)
    if outcome.trace_id:
        record["trace_id"] = outcome.trace_id
    else:
        record.pop("trace_id", None)
    if outcome.error:
        record["error"] = outcome.error
    else:
        record.pop("error", None)
    if outcome.attempts > 1:
        record["attempts"] = outcome.attempts
    else:
        record.pop("attempts", None)


def _write_back(path: Path, records: list[dict]) -> None:
    """Rewrite the answers file atomically: complete content first, one replace."""
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix.lower() == ".xlsx":
        import openpyxl

        header: list[str] = []
        for record in records:
            header += [k for k in record if k not in header]
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.append(header)
        for record in records:
            sheet.append([batch_request._cell(record.get(k)) for k in header])
        book.save(tmp)
    else:
        tmp.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in records) + "\n",
            encoding="utf-8")
    os.replace(tmp, path)


def classify(error: str) -> str:
    """One bucket per remedy: timeouts want bigger knobs, the rest want luck."""
    if error.startswith("TimeoutError"):
        return "timeout — raise CHILLRAG_ANSWER_TIMEOUT_S / tune CHILLRAG_MAX_TOOL_CALLS, rerun with --retries 0"
    if "connection" in error.lower() or "apiconnection" in error.lower():
        return "network — transient; --retries (the default) should absorb it"
    return "other — inspect the error field before rerunning"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("answers", type=Path,
                        help="the answers file from a batch run (.jsonl or .xlsx), updated in place")
    parser.add_argument("-c", "--concurrency", type=int, default=2,
                        help="parallel episodes (default 2)")
    parser.add_argument("--retries", type=int, default=2,
                        help="retries per row on failure — network blips only (default 2)")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--folder", help="corpus for rows that name none")
    parser.add_argument("--user", help="identity for rows that name none")
    parser.add_argument("--source", help="surface for rows that name none")
    parser.add_argument("--lang", help="prompt language for rows that name none")
    parser.add_argument("--mode", choices=("agent", "oneshot"),
                        help="answer mode for rows that name none")
    parser.add_argument("--instructions", help="system-prompt requirements for rows that name none")
    parser.add_argument("--output-instruction",
                        help="per-question output contract for rows that name none")
    args = parser.parse_args(argv)

    records = _read_records(args.answers)
    targets = _failed(records)
    if not targets:
        print(f"nothing to retry in {args.answers}")
        return 0
    print(f"{len(targets)} row(s) to re-answer → {args.answers}")

    jobs = batch_request._plan([row for _, row in targets], args)
    tool = batch_request._load_tool({job.folder for job in jobs})
    orig_row = [rec.get("index", i) + 1 for i, (rec, _) in enumerate(targets)]
    done = [0]

    def on_outcome(outcome) -> None:
        done[0] += 1
        mark = {"ok": "✓", "empty": "∅", "error": "✗"}[outcome.status]
        print(f"[{done[0]}/{len(jobs)}] {mark} row {orig_row[outcome.index]}: "
              f"{outcome.status} ({outcome.duration_s:.1f}s)")

    outcomes = asyncio.run(batch_request.run_batch(
        jobs, tool, concurrency=args.concurrency, retries=max(0, args.retries),
        on_outcome=on_outcome))

    for (record, _), outcome in zip(targets, outcomes):
        _patch(record, outcome)
    _write_back(args.answers, records)

    left = sum(o.status == "error" for o in outcomes)
    print(f"{len(outcomes) - left} recovered · {left} still failing → {args.answers}")
    for (_, _), outcome in zip(targets, outcomes):
        if outcome.status == "error":
            print(f"  {classify(outcome.error)}")
    return 1 if left else 0


if __name__ == "__main__":
    raise SystemExit(main())
