#!/usr/bin/env python3
"""Create an opt-in Vision-OPD dataset with original image and bbox metadata."""
import argparse, json
from pathlib import Path

import datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--input-parquet", required=True)
    ap.add_argument("--output-parquet", required=True)
    args = ap.parse_args()
    jsonl_path = Path(args.input_jsonl).resolve()
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    table = datasets.load_dataset("parquet", data_files=args.input_parquet)["train"]
    if len(rows) != len(table):
        raise ValueError(f"row count mismatch: jsonl={len(rows)} parquet={len(table)}")
    bboxes = []
    for i, row in enumerate(rows):
        if "bbox" not in row:
            raise KeyError(f"row {i} missing bbox")
        bbox = row["bbox"]
        if len(bbox) != 4:
            raise ValueError(f"row {i} bbox must have four coordinates: {bbox}")
        bboxes.append([float(x) for x in bbox])
        if any(v < 0 for v in bboxes[-1]) or bboxes[-1][2] <= bboxes[-1][0] or bboxes[-1][3] <= bboxes[-1][1]:
            raise ValueError(f"row {i} bbox must satisfy 0 <= x1 < x2 and 0 <= y1 < y2: {bbox}")
    table = table.add_column("bbox", bboxes)
    Path(args.output_parquet).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_parquet)
    print(f"wrote {len(table)} rows to {args.output_parquet}")


if __name__ == "__main__":
    main()
