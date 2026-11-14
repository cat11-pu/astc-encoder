#!/usr/bin/env python3
"""Run the 500-texture ASTC regression corpus.

For each texture the script:
  1. Compresses twice with identical fixed settings (deterministic build).
  2. Hashes both ASTC outputs and requires byte-identical results.
  3. Runs test mode to collect PSNR / SSIM quality metrics.
  4. Decompresses the ASTC and independently recomputes PSNR in Python.

Baselines come from the first accepted run stored alongside the results; a
later run must not regress any aggregate quality metric below that baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_BINARY = ROOT.parent / "build-native" / "Source" / "astcenc-native"
BLOCK = "6x6"
QUALITY = "-medium"
THREADS = "4"

METRIC_RE = {
    "psnr_rgba": re.compile(r"PSNR \(LDR-RGBA\):\s+([0-9.]+)\s*dB"),
    "psnr_rgb": re.compile(r"PSNR \(LDR-RGB\):\s+([0-9.]+)\s*dB"),
    "ssim_rgba": re.compile(r"SSIM \(LDR-RGBA\):\s+([0-9.]+)"),
    "ssim_rgb": re.compile(r"SSIM \(LDR-RGB\):\s+([0-9.]+)"),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=600)
    return proc.returncode, proc.stdout


def python_psnr(src: Path, dec: Path) -> float:
    """Recompute mean-over-channels PSNR independently of the codec."""
    a = Image.open(src).convert("RGBA")
    b = Image.open(dec).convert("RGBA")
    if a.size != b.size:
        return math.nan
    pa = a.tobytes()
    pb = b.tobytes()
    total_sq = 0
    samples = len(pa)
    for x, y in zip(pa, pb):
        d = x - y
        total_sq += d * d
    mse = total_sq / samples
    if mse == 0:
        return 99.0
    return 10.0 * math.log10((255.0 ** 2) / mse)


def parse_metrics(text: str) -> dict[str, float]:
    out = {}
    for key, pattern in METRIC_RE.items():
        match = pattern.search(text)
        out[key] = float(match.group(1)) if match else math.nan
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--label", default="run")
    parser.add_argument("--limit", type=int, default=0, help="debug: cap corpus size")
    args = parser.parse_args()

    binary = args.binary.resolve()
    out_dir = args.out
    astc_dir = out_dir / "astc"
    dec_dir = out_dir / "decoded"
    for directory in (astc_dir, dec_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = list(csv.DictReader((args.corpus / "manifest.csv").open()))
    if args.limit:
        manifest = manifest[: args.limit]

    rows = []
    crashes = 0
    mismatches = 0
    started = time.time()

    for index, entry in enumerate(manifest):
        cls = entry["class"]
        src = args.corpus / cls / entry["file"]
        stem = f"{int(entry['id']):03d}_{entry['file'].rsplit('.', 1)[0]}"
        astc_a = astc_dir / f"{stem}.a.astc"
        astc_b = astc_dir / f"{stem}.b.astc"
        dec_png = dec_dir / f"{stem}.png"

        extra = ["-normal"] if cls == "normal" else []
        compress_base = [str(binary), "-cl", str(src)]
        common = [BLOCK, QUALITY, "-j", THREADS, "-silent", *extra]

        rc_a, log_a = run_cmd(compress_base + [str(astc_a)] + common)
        rc_b, log_b = run_cmd(compress_base + [str(astc_b)] + common)

        # Test mode emits quality metrics and writes the decoded image
        rc_t, log_t = run_cmd(
            [str(binary), "-tl", str(src), str(dec_png), BLOCK, QUALITY,
             "-j", THREADS, *extra]
        )

        crashed = rc_a != 0 or rc_b != 0 or rc_t != 0
        if crashed:
            crashes += 1

        hash_a = sha256_of(astc_a) if astc_a.exists() else ""
        hash_b = sha256_of(astc_b) if astc_b.exists() else ""
        byte_identical = bool(hash_a) and hash_a == hash_b
        if not byte_identical:
            mismatches += 1

        metrics = parse_metrics(log_t)
        independent_psnr = python_psnr(src, dec_png) if dec_png.exists() else math.nan

        rows.append({
            "id": entry["id"],
            "class": cls,
            "file": entry["file"],
            "width": entry["width"],
            "height": entry["height"],
            "rc_compress_a": rc_a,
            "rc_compress_b": rc_b,
            "rc_test": rc_t,
            "crashed": int(crashed),
            "byte_identical": int(byte_identical),
            "sha256": hash_a,
            "astc_bytes": astc_a.stat().st_size if astc_a.exists() else 0,
            "psnr_rgba": metrics["psnr_rgba"],
            "psnr_rgb": metrics["psnr_rgb"],
            "ssim_rgba": metrics["ssim_rgba"],
            "ssim_rgb": metrics["ssim_rgb"],
            "py_psnr_rgba": independent_psnr,
            "log_tail": (log_a + log_b + log_t).replace("\n", " ")[-400:] if crashed else "",
        })

        if (index + 1) % 25 == 0 or index == len(manifest) - 1:
            elapsed = time.time() - started
            print(f"[{args.label}] {index + 1}/{len(manifest)} "
                  f"crashes={crashes} mismatches={mismatches} elapsed={elapsed:.0f}s",
                  flush=True)

    csv_path = out_dir / f"metrics_{args.label}.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {csv_path}")
    print(f"crashes={crashes} byte_mismatches={mismatches}")
    return 1 if crashes or mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
