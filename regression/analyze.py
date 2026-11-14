#!/usr/bin/env python3
"""Analyze two regression runs: byte identity, baseline stats, worst samples."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load(label: str) -> list[dict]:
    path = RESULTS / label / f"metrics_{label}.csv"
    with path.open() as fh:
        return list(csv.DictReader(fh))


def to_float(value: str) -> float:
    return float(value) if value not in ("", "nan") else math.nan


def sha256_tree(directory: Path) -> dict[str, str]:
    out = {}
    for path in sorted(directory.rglob("*.astc")):
        out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return math.nan
    idx = min(len(sorted_values) - 1, int(round(pct / 100.0 * (len(sorted_values) - 1))))
    return sorted_values[idx]


def main() -> None:
    run1 = load("run1")
    run2 = load("run2")
    assert len(run1) == len(run2) == 500

    # 1. Cross-run byte identity of every ASTC artifact
    tree1 = sha256_tree(RESULTS / "run1" / "astc")
    tree2 = sha256_tree(RESULTS / "run2" / "astc")
    cross_diffs = [name for name in tree1 if tree1.get(name) != tree2.get(name)]
    within_a = sum(int(r["byte_identical"]) for r in run1)
    within_b = sum(int(r["byte_identical"]) for r in run2)

    # 2. Metric equality between runs
    metric_diffs = []
    for a, b in zip(run1, run2):
        for key in ("psnr_rgba", "psnr_rgb", "ssim_rgba", "ssim_rgb"):
            if to_float(a[key]) != to_float(b[key]):
                metric_diffs.append((a["id"], key, a[key], b[key]))

    metrics = ("psnr_rgba", "psnr_rgb", "ssim_rgba", "ssim_rgb", "py_psnr_rgba")
    classes = sorted({r["class"] for r in run1})

    def stats(rows: list[dict], key: str) -> dict:
        vals = sorted(to_float(r[key]) for r in rows if not math.isnan(to_float(r[key])))
        finite = [v for v in vals if v < 900]  # exclude lossless sentinel 999
        return {
            "mean": sum(vals) / len(vals),
            "min": vals[0],
            "p01": percentile(vals, 1),
            "p05": percentile(vals, 5),
            "median": percentile(vals, 50),
            "finite_min": finite[0] if finite else math.nan,
            "lossless": len(vals) - len(finite),
        }

    overall = {key: stats(run1, key) for key in metrics}
    by_class = {
        cls: {key: stats([r for r in run1 if r["class"] == cls], key) for key in metrics}
        for cls in classes
    }

    # 3. Independent PSNR vs codec-reported PSNR agreement
    agree_max_err = 0.0
    for r in run1:
        reported = to_float(r["psnr_rgba"])
        independent = to_float(r["py_psnr_rgba"])
        if reported < 900:
            agree_max_err = max(agree_max_err, abs(reported - independent))

    # 4. Worst samples (finite PSNR only; lossless sentinel excluded)
    worst_psnr = sorted(
        (r for r in run1 if to_float(r["psnr_rgba"]) < 900),
        key=lambda r: to_float(r["psnr_rgba"]),
    )[:10]
    worst_ssim = sorted(run1, key=lambda r: to_float(r["ssim_rgba"]))[:10]

    # 5. Crash / return code accounting
    bad_rc = [
        (r["id"], r["file"], r["rc_compress_a"], r["rc_compress_b"], r["rc_test"])
        for r in run1 + run2
        if r["rc_compress_a"] != "0" or r["rc_compress_b"] != "0" or r["rc_test"] != "0"
    ]

    summary = {
        "sample_count": len(run1),
        "classes": {cls: sum(1 for r in run1 if r["class"] == cls) for cls in classes},
        "crashes_total": len(bad_rc),
        "within_run_identical": {"run1": within_a, "run2": within_b},
        "cross_run_astc_diffs": len(cross_diffs),
        "cross_run_metric_diffs": len(metric_diffs),
        "codec_vs_python_psnr_max_abs_err": agree_max_err,
        "overall": overall,
        "by_class": by_class,
        "worst_psnr": [
            {
                "id": r["id"], "class": r["class"], "file": r["file"],
                "size": f"{r['width']}x{r['height']}",
                "psnr_rgba": to_float(r["psnr_rgba"]),
                "psnr_rgb": to_float(r["psnr_rgb"]),
                "ssim_rgba": to_float(r["ssim_rgba"]),
                "ssim_rgb": to_float(r["ssim_rgb"]),
            }
            for r in worst_psnr
        ],
        "worst_ssim": [
            {
                "id": r["id"], "class": r["class"], "file": r["file"],
                "size": f"{r['width']}x{r['height']}",
                "psnr_rgba": to_float(r["psnr_rgba"]),
                "ssim_rgba": to_float(r["ssim_rgba"]),
            }
            for r in worst_ssim
        ],
    }

    out = RESULTS / "analysis.json"
    out.write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False)[:4000])
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
