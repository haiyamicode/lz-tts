#!/usr/bin/env python3
"""Score a Sparrow evaluation manifest with UTMOS."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from utmos import Score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--score-references", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    scorer = Score()
    reference_cache: dict[str, float] = {}

    for index, row in enumerate(rows):
        row["utmos"] = float(scorer.calculate_wav_file(row["path"]))
        if args.score_references:
            reference = row["reference"]
            if reference not in reference_cache:
                reference_cache[reference] = float(scorer.calculate_wav_file(reference))
            row["reference_utmos"] = reference_cache[reference]
        print(
            f"{index + 1}/{len(rows)} {row['language']} utmos={row['utmos']:.3f}",
            flush=True,
        )

    scored_path = manifest_path.with_name("manifest_utmos.json")
    scored_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = manifest_path.with_name("utmos_scores.csv")
    fields = [
        "group",
        "language",
        "source",
        "utmos",
        "reference_utmos",
        "generated_duration",
        "reference_duration",
        "duration_ratio",
        "path",
        "reference",
        "text",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_samples": len(rows),
        "utmos_mean": statistics.fmean(row["utmos"] for row in rows),
        "utmos_median": statistics.median(row["utmos"] for row in rows),
        "utmos_min": min(row["utmos"] for row in rows),
        "utmos_max": max(row["utmos"] for row in rows),
        "by_group": {},
    }
    for group in sorted({row["group"] for row in rows}):
        values = [row["utmos"] for row in rows if row["group"] == group]
        summary["by_group"][group] = {
            "n": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        }
    manifest_path.with_name("utmos_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
