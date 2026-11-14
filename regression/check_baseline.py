#!/usr/bin/env python3
"""Compare a run against the frozen baseline and fail on quality regression."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
METRICS = ("psnr_rgba", "psnr_rgb", "ssim_rgba", "ssim_rgb")


def fnum(value: str) -> float:
    return float(value) if value not in ("", "nan") else math.nan


def load_rows(label: str) -> list[dict]:
    with (RESULTS / label / f"metrics_{label}.csv").open() as fh:
        return list(csv.DictReader(fh))


def aggregate(rows: list[dict]) -> dict:
    out = {}
    for key in METRICS:
        vals = [fnum(r[key]) for r in rows if not math.isnan(fnum(r[key]))]
        finite = [v for v in vals if v < 900]
        out[key] = {
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "finite_min": min(finite),
            "p05": sorted(vals)[max(0, int(0.05 * (len(vals) - 1)))],
        }
    return out


def main() -> int:
    baseline = json.loads((RESULTS / "baseline.json").read_text())
    label = "run2"
    rows = load_rows(label)
    current = aggregate(rows)

    failures = []
    # Every aggregate quality indicator must be at least the baseline value.
    for key in METRICS:
        for stat in ("mean", "min", "finite_min", "p05"):
            base_val = baseline["aggregate"][key][stat]
            cur_val = current[key][stat]
            if cur_val + 1e-9 < base_val:
                failures.append(f"{key}.{stat}: baseline={base_val:.4f} current={cur_val:.4f}")

    # Per-sample: no individual texture may fall more than 0.1 dB / 0.002 SSIM
    # below its own baseline measurement.
    base_rows = {r["id"]: r for r in load_rows("run1")}
    tolerances = {"psnr_rgba": 0.1, "psnr_rgb": 0.1, "ssim_rgba": 0.002, "ssim_rgb": 0.002}
    for row in rows:
        ref = base_rows[row["id"]]
        for key, tol in tolerances.items():
            if fnum(row[key]) + tol < fnum(ref[key]):
                failures.append(
                    f"sample {row['id']} {key}: baseline={fnum(ref[key]):.4f} "
                    f"current={fnum(row[key]):.4f}"
                )

    report = {
        "label": label,
        "samples": len(rows),
        "aggregate_current": current,
        "failures": failures,
        "passed": not failures,
    }
    (RESULTS / "baseline_check.json").write_text(json.dumps(report, indent=2))
    if failures:
        print("BASELINE CHECK FAILED:")
        for item in failures[:20]:
            print(" -", item)
        return 1
    print(f"Baseline check passed: {len(rows)} samples, all metrics >= baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
