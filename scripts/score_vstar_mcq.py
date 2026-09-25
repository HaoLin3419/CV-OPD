#!/usr/bin/env python3
"""Deterministically score VStar letter answers, with optional paired analysis."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ANSWER_PATTERNS = (
    re.compile(r"<answer>\s*\(?\s*([A-D])\s*\)?\s*</answer>", re.I),
    re.compile(r"(?:answer|option|choice)\s*(?:is|:|=)?\s*\(?\s*([A-D])\s*\)?", re.I),
    re.compile(r"^\s*\(?\s*([A-D])\s*\)?(?:[\s.!,:;-]|$)", re.I),
)
PARENTHESIZED_OPTION = re.compile(r"\(\s*([A-D])\s*\)", re.I)
MARKDOWN_OPTION = re.compile(r"\*\*\s*([A-D])\s*\*\*", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--baseline-answers", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_letter(raw: Any) -> str | None:
    text = str(raw or "").strip()
    think_end = text.rfind("</think>")
    if think_end >= 0:
        text = text[think_end + len("</think>") :].strip()
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    # Models often explain first and finish with "Correct answer: **(C) ...**".
    matches = PARENTHESIZED_OPTION.findall(text)
    if matches:
        return matches[-1].upper()
    matches = MARKDOWN_OPTION.findall(text)
    if matches:
        return matches[-1].upper()
    return None


def evaluate(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, bool]]:
    groups: dict[str, list[bool]] = defaultdict(list)
    correctness: dict[str, bool] = {}
    unparsed: list[str] = []
    for record in records:
        question_id = str(record["question_id"])
        prediction = extract_letter(record.get("model_answer"))
        target = str(record.get("response", "")).strip().upper()
        correct = prediction == target
        correctness[question_id] = correct
        groups["overall"].append(correct)
        groups[f"category/{record.get('category', 'unknown')}"] .append(correct)
        box_count = len(record.get("bbox_xyxy") or [])
        groups[f"target_box_count/{box_count}"].append(correct)
        if prediction is None:
            unparsed.append(question_id)
    stats = {
        name: {
            "correct": sum(values),
            "total": len(values),
            "accuracy": sum(values) / len(values) if values else 0.0,
        }
        for name, values in sorted(groups.items())
    }
    return {"groups": stats, "unparsed_question_ids": unparsed}, correctness


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.answers)
    if len(records) != 191:
        raise SystemExit(f"expected 191 answer records, found {len(records)} in {args.answers}")
    result, noisy_correct = evaluate(records)
    result["answers"] = str(args.answers.resolve())

    if args.baseline_answers:
        baseline_records = load_jsonl(args.baseline_answers)
        if len(baseline_records) != 191:
            raise SystemExit(
                f"expected 191 baseline records, found {len(baseline_records)}"
            )
        baseline_result, baseline_correct = evaluate(baseline_records)
        if baseline_correct.keys() != noisy_correct.keys():
            raise SystemExit("baseline and perturbed question_id sets differ")
        transitions = defaultdict(int)
        for question_id in noisy_correct:
            before = baseline_correct[question_id]
            after = noisy_correct[question_id]
            transitions[f"{'correct' if before else 'wrong'}_to_{'correct' if after else 'wrong'}"] += 1
        result["baseline"] = baseline_result
        result["paired_transitions"] = dict(sorted(transitions.items()))
        result["accuracy_delta"] = (
            result["groups"]["overall"]["accuracy"]
            - baseline_result["groups"]["overall"]["accuracy"]
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
