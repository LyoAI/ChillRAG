"""Build a retry input from a batch answers file.

Reads ``<input>.answers.jsonl``, keeps the rows that did not answer —
``status: "error"``, plus ``empty`` with ``--include-empty`` — and strips the
output-only fields, leaving exactly what a re-run needs: the question and the
row's own metadata (folder, user, …), so the retry answers the same question
on the same corpus under the same identity. Writes ``<stem>.retry.jsonl``
next to the answers file and prints the failure breakdown, so the follow-up
``batch_request`` call is one copy-paste.

    python -m chillrag.utils.retry_failed runs/q.answers.jsonl
    python -m chillrag.utils.retry_failed q.answers.jsonl --include-empty -c 4

Extraction only — it never re-answers by itself. Timeouts dominate the
breakdown? Raise ``CHILLRAG_MAX_TOOL_CALLS`` / ``CHILLRAG_ANSWER_TIMEOUT_S``
and pass ``--retries 0``; otherwise let ``--retries`` absorb the blips.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

#: Fields the batch driver adds to a row when it reports an outcome. The rest
#: of the record is the input row as it was read, and rides along on a retry.
OUTPUT_FIELDS = frozenset(
    {"index", "answer", "status", "duration_s", "trace_id", "error", "attempts"})


def failed_rows(records: list[dict], include_empty: bool = False) -> list[dict]:
    """The unanswered rows as clean input rows, in original input order."""
    bad = [r for r in records
           if r.get("status") == "error" or (include_empty and r.get("status") == "empty")]
    bad.sort(key=lambda r: r.get("index", 0))
    return [{k: v for k, v in r.items() if k not in OUTPUT_FIELDS} for r in bad]


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
    parser.add_argument("answers", type=Path, help="the .answers.jsonl from a batch run")
    parser.add_argument("--include-empty", action="store_true",
                        help="also retry rows whose answer came back empty")
    parser.add_argument("--out", type=Path,
                        help="retry input path (default: <stem>.retry.jsonl alongside)")
    args = parser.parse_args(argv)

    if args.answers.suffix.lower() != ".jsonl":
        raise SystemExit(f"expected a .jsonl answers file, got {args.answers.name!r}")
    records = [json.loads(line)
               for line in args.answers.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = failed_rows(records, include_empty=args.include_empty)
    if not rows:
        print(f"nothing to retry in {args.answers}")
        return 0

    out_path = args.out or args.answers.with_name(
        args.answers.stem.removesuffix(".answers") + ".retry.jsonl")
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + "\n",
        encoding="utf-8")

    errors = [r.get("error", "") for r in records if r.get("status") == "error"]
    print(f"{len(rows)} row(s) to retry → {out_path}")
    for kind, n in Counter(classify(e) for e in errors).most_common():
        print(f"  {n:4d}  {kind}")
    print(f"rerun: python -m chillrag.batch_request {out_path} -c 2"
          + ("  # see the timeout note above" if "timeout" in " ".join(errors).lower() else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
