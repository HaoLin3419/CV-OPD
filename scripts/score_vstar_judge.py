#!/usr/bin/env python3
"""Create a fixed-sigma VStar score file from canonical Yes/No Judge output."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--protocol", default="VAD GPT-OSS-120B Judge + single-letter normalization + noncanonical-as-No")
    args = parser.parse_args()

    records = json.loads(args.judge_json.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 191:
        raise SystemExit(f"expected 191 Judge records, found {len(records) if isinstance(records, list) else 'non-list'}")

    groups: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            raise SystemExit("Judge output contains a non-object record")
        judge = str(record.get("judge", "")).strip()
        if judge not in {"Yes", "No"}:
            raise SystemExit(f"noncanonical Judge label: {judge!r}")
        correct = judge == "Yes"
        groups["overall"].append(correct)
        groups[f"category/{record.get('category', 'unknown')}"] .append(correct)
        groups[f"target_box_count/{len(record.get('bbox_xyxy') or [])}"].append(correct)

    stats = {
        name: {
            "correct": sum(values),
            "total": len(values),
            "accuracy": sum(values) / len(values) if values else 0.0,
        }
        for name, values in sorted(groups.items())
    }
    result: dict[str, Any] = {
        "groups": stats,
        "unparsed_question_ids": [],
        "answers": str(args.answers.resolve()),
        "judge": str(args.judge_json.resolve()),
        "protocol": args.protocol,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
