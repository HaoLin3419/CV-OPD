"""Binary exact-match reward for the VStar multiple-choice training records.

The parser intentionally mirrors ``scripts/score_vstar_mcq.py`` so training
rewards and post-training evaluation use the same answer extraction policy.
"""

from __future__ import annotations

import re
from typing import Any


ANSWER_PATTERNS = (
    re.compile(r"<answer>\s*\(?\s*([A-D])\s*\)?\s*</answer>", re.I),
    re.compile(r"(?:answer|option|choice)\s*(?:is|:|=)?\s*\(?\s*([A-D])\s*\)?", re.I),
    re.compile(r"^\s*\(?\s*([A-D])\s*\)?(?:[\s.!,:;-]|$)", re.I),
)
PARENTHESIZED_OPTION = re.compile(r"\(\s*([A-D])\s*\)", re.I)
MARKDOWN_OPTION = re.compile(r"\*\*\s*([A-D])\s*\*\*", re.I)


def extract_letter(raw: Any) -> str | None:
    text = str(raw or "").strip()
    think_end = text.rfind("</think>")
    if think_end >= 0:
        text = text[think_end + len("</think>") :].strip()
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    matches = PARENTHESIZED_OPTION.findall(text)
    if matches:
        return matches[-1].upper()
    matches = MARKDOWN_OPTION.findall(text)
    if matches:
        return matches[-1].upper()
    return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return reward 1 for an exact VStar option match, otherwise 0."""

    prediction = extract_letter(solution_str)
    target = str(ground_truth or "").strip().upper()
    exact_match = prediction is not None and prediction == target
    return {
        "score": float(exact_match),
        "exact_match": bool(exact_match),
        "parsed_answer": prediction,
        "ground_truth": target,
    }
