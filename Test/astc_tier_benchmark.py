#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ----------------------------------------------------------------------------
# Copyright 2026 Arm Limited
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
# ----------------------------------------------------------------------------

"""
Benchmark and conformance harness for the three quality tiers.

This script validates the quality tier guarantees:

    * All tiers emit standard ASTC data that the reference decoder accepts.
    * Decoding is deterministic; decoding the same data twice gives
      bit-identical pixels.
    * ASTCENC_TIER_FAST encodes at least 30% faster than ASTCENC_TIER_MEDIUM.
    * ASTCENC_TIER_HIGH achieves at least 1 dB higher PSNR than
      ASTCENC_TIER_MEDIUM.

Usage:
    python3 astc_tier_benchmark.py <astcenc-binary> [--outdir DIR] [--repeats N]

Results are written to <outdir>/tier_benchmark.csv and a human-readable
summary is written to <outdir>/tier_report.md.
"""

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
import tempfile

# Tier name -> (CLI quality argument, block size, bits per texel)
TIERS = {
    "fast":   ("-tier:fast",   "8x8", 2.00),
    "medium": ("-tier:medium", "6x6", 3.56),
    "high":   ("-tier:high",   "4x4", 8.00),
}

# Test image sets: (directory, CLI compress mode, CLI test mode)
IMAGE_SETS = [
    ("LDR-RGB", "-cl", "-tl"),
    ("LDR-RGBA", "-cl", "-tl"),
    ("LDRS-RGBA", "-cs", "-ts"),
]

# Constraint thresholds
FAST_SPEEDUP_MIN = 1.30   # fast must be >= 30% faster than medium
HIGH_PSNR_GAIN_MIN = 1.0  # high must be >= 1 dB above medium

TIME_RE = re.compile(r"Coding time:\s+([0-9.]+) s")
PSNR_RGBA_RE = re.compile(r"PSNR \(LDR-RGBA\):\s+([0-9.]+) dB")
PSNR_RGB_RE = re.compile(r"PSNR \(LDR-RGB\):\s+([0-9.]+) dB")


def find_images(test_root):
    """Find the benchmark input images."""
    images = []
    small_dir = os.path.join(test_root, "Test", "Images", "Small")
    for subdir, mode, test_mode in IMAGE_SETS:
        set_dir = os.path.join(small_dir, subdir)
        for name in sorted(os.listdir(set_dir)):
            if name.endswith(".png"):
                images.append(
                    (subdir, mode, test_mode, os.path.join(set_dir, name)))
    return images


def run_test(binary, test_mode, image, tier_arg, out_png):
    """Run one test-mode compress+decompress, returning (time_s, psnr_db)."""
    # The block size argument is ignored when a quality tier is used
    cmd = [binary, test_mode, image, out_png, "6x6", tier_arg, "-j", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = result.stdout
    time_match = TIME_RE.search(output)
    psnr_match = PSNR_RGBA_RE.search(output) or PSNR_RGB_RE.search(output)
    if not time_match or not psnr_match:
        raise RuntimeError("Failed to parse encoder output:\n" + output)
    return float(time_match.group(1)), float(psnr_match.group(1))


def run_compress(binary, mode, image, tier_arg, out_astc):
    """Run one compression to produce an ASTC file."""
    cmd = [binary, mode, image, out_astc, "6x6", tier_arg, "-j", "1",
           "-silent"]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


def decode_sha256(binary, astc_path, out_png):
    """Decompress an ASTC file and return the SHA-256 of the decoded PNG."""
    cmd = [binary, "-dl", astc_path, out_png]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    with open(out_png, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", help="Path to the astcenc binary")
    parser.add_argument("--outdir", default="artifacts/tier_data",
                        help="Output directory for result data")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Encode repetitions per case; min time is used")
    args = parser.parse_args()

    binary = os.path.abspath(args.binary)
    test_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images = find_images(test_root)
    if not images:
        print("ERROR: no test images found", file=sys.stderr)
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    decode_failures = []
    determinism_failures = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for subdir, mode, test_mode, image in images:
            name = os.path.basename(image)
            for tier, (tier_arg, block, _bpp) in TIERS.items():
                astc_path = os.path.join(
                    tmpdir, "%s_%s.astc" % (name, tier))
                out_png = os.path.join(tmpdir, "test_out.png")

                # Time repeated encodes; keep the minimum as the estimate
                times = []
                psnr = None
                for _ in range(args.repeats):
                    coding_time, run_psnr = run_test(
                        binary, test_mode, image, tier_arg, out_png)
                    times.append(coding_time)
                    if psnr is None:
                        psnr = run_psnr

                # Produce the compressed file for the decode checks
                run_compress(binary, mode, image, tier_arg, astc_path)

                # Standard decoder must accept the output, and two
                # decodes must be bit-identical
                try:
                    png1 = os.path.join(tmpdir, "dec1.png")
                    png2 = os.path.join(tmpdir, "dec2.png")
                    hash1 = decode_sha256(binary, astc_path, png1)
                    hash2 = decode_sha256(binary, astc_path, png2)
                    if hash1 != hash2:
                        determinism_failures.append((name, tier))
                except subprocess.CalledProcessError:
                    decode_failures.append((name, tier))

                rows.append({
                    "image_set": subdir,
                    "image": name,
                    "tier": tier,
                    "block_size": block,
                    "coding_time_s": "%.6f" % min(times),
                    "psnr_db": "%.4f" % psnr,
                })
                print("  %-12s %-14s %-6s  %8.4f s  %7.4f dB"
                      % (subdir, name, tier, min(times), psnr))

    csv_path = os.path.join(args.outdir, "tier_benchmark.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate per tier and check the tier constraints
    stats = {}
    for tier in TIERS:
        sel = [r for r in rows if r["tier"] == tier]
        stats[tier] = (
            sum(float(r["coding_time_s"]) for r in sel) / len(sel),
            sum(float(r["psnr_db"]) for r in sel) / len(sel),
        )

    med_time, med_psnr = stats["medium"]
    fast_time, _ = stats["fast"]
    _, high_psnr = stats["high"]

    speedup = med_time / fast_time
    psnr_gain = high_psnr - med_psnr
    speed_ok = speedup >= FAST_SPEEDUP_MIN
    psnr_ok = psnr_gain >= HIGH_PSNR_GAIN_MIN
    all_ok = (speed_ok and psnr_ok
              and not decode_failures and not determinism_failures)

    # Write the human-readable report
    lines = ["# Quality tier benchmark report", ""]
    lines.append("Binary: `%s`" % os.path.basename(binary))
    lines.append("Images: %d  Repeats: %d (min time), single-threaded"
                 % (len(images), args.repeats))
    lines.append("")
    lines.append("| Tier | Block size | bpp | Mean coding time (s) | "
                 "Mean PSNR (dB) |")
    lines.append("|------|------------|-----|----------------------|"
                 "----------------|")
    for tier, (_arg, block, bpp) in TIERS.items():
        lines.append("| %s | %s | %.2f | %.4f | %.4f |"
                     % (tier, block, bpp, stats[tier][0], stats[tier][1]))
    lines.append("")
    lines.append("## Constraint checks")
    lines.append("")
    lines.append("| Check | Requirement | Measured | Result |")
    lines.append("|-------|-------------|----------|--------|")
    lines.append("| fast encode speed vs medium | >= 1.30x | %.2fx | %s |"
                 % (speedup, "PASS" if speed_ok else "FAIL"))
    lines.append("| high PSNR gain vs medium | >= +1.00 dB | %+.2f dB | %s |"
                 % (psnr_gain, "PASS" if psnr_ok else "FAIL"))
    lines.append("| standard decoder accepts all outputs | yes | %s | %s |"
                 % ("yes" if not decode_failures else decode_failures,
                    "PASS" if not decode_failures else "FAIL"))
    lines.append("| repeated decodes bit-identical | yes | %s | %s |"
                 % ("yes" if not determinism_failures
                    else determinism_failures,
                    "PASS" if not determinism_failures else "FAIL"))

    report_path = os.path.join(args.outdir, "tier_report.md")
    with open(report_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")

    print()
    print("\n".join(lines))
    print()
    print("Wrote %s and %s" % (csv_path, report_path))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
