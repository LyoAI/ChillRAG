"""Extract tagged answers out of a batch answers file, in place.

A batch run whose ``output_instruction`` asked for the answer inside a tag
pair (``<answer>...</answer>``) leaves the wrapper in the answers column;
scoring wants the bare value. This rewrites the answers file, replacing each
wrapped answer with its extracted body — rows without the tag are left
untouched and counted, so the script is idempotent and safe to run twice.

Caller-side tooling, deliberately outside the ``chillrag`` package: the tag
contract belongs to the caller that asked for it.

    python scripts/extract_answer.py runs/q.answers.jsonl
    python scripts/extract_answer.py runs/q.xlsx --tag answer --field answer
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from chillrag.utils.answers import read_answers, write_answers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("answers", type=Path, help="the answers file, updated in place")
    parser.add_argument("--tag", default="answer", help="wrapper tag name (default: answer)")
    parser.add_argument("--field", default="answer", help="column holding the wrapped text")
    args = parser.parse_args(argv)

    rx = re.compile(rf"<{re.escape(args.tag)}>(.*?)</{re.escape(args.tag)}>", re.DOTALL)
    rows = read_answers(args.answers)
    extracted = untagged = 0
    for row in rows:
        text = str(row.get(args.field) or "")
        match = rx.search(text)
        if match:
            row[args.field] = match.group(1).strip()
            extracted += 1
        elif text.strip():
            untagged += 1
    write_answers(args.answers, rows)
    print(f"{extracted} row(s) extracted · {untagged} without <{args.tag}> "
          f"(left as-is) → {args.answers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
