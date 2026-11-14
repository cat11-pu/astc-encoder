#!/usr/bin/env python3
"""Generate a fixed corpus of 500 textures across six texture classes.

Classes: solid, gradient, hardedge, noise, alpha, normal.

All generation is deterministic: a fixed PRNG seed and fixed parameter tables
make the corpus byte-reproducible across runs and machines.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

from PIL import Image


# Deterministic LCG (no numpy dependency)
class Rng:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        # MMIX constants
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.state

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * (self.next_u64() / 18446744073709551616.0)

    def integer(self, lo: int, hi: int) -> int:
        return lo + self.next_u64() % (hi - lo + 1)

    def channel(self) -> int:
        return self.next_u64() & 0xFF


# Base palette of sizes; per-index multipliers keep every dimension <= 512.
SIZES = [
    (16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (80, 80),
    (96, 96), (128, 128), (160, 160), (192, 192), (256, 256),
    (32, 16), (64, 32), (128, 64), (256, 128),
    (16, 32), (32, 64), (64, 128), (128, 256),
    (40, 24), (72, 56), (120, 88), (200, 152),
]
MULTIPLIERS = (1, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2)

CLASSES = ("solid", "gradient", "hardedge", "noise", "alpha", "normal")
COUNTS = {"solid": 83, "gradient": 83, "hardedge": 84, "noise": 84, "alpha": 83, "normal": 83}


def size_for(index: int) -> tuple[int, int]:
    base = SIZES[index % len(SIZES)]
    scale = MULTIPLIERS[index % len(MULTIPLIERS)]
    width = min(512, base[0] * scale)
    height = min(512, base[1] * scale)
    # Round down to a multiple of 4, which is the ASTC dimension granularity
    width = max(4, width - width % 4)
    height = max(4, height - height % 4)
    return width, height


def gen_solid(rng: Rng, w: int, h: int) -> Image.Image:
    color = (rng.channel(), rng.channel(), rng.channel())
    img = Image.new("RGB", (w, h), color)
    # A minority carry subtle row/column banding, still "solid" class
    if rng.integer(0, 3) == 0:
        px = img.load()
        delta = rng.integer(1, 6)
        for y in range(h):
            tint = (y * delta) & 0xFF
            for x in range(w):
                px[x, y] = (
                    (color[0] + tint) & 0xFF,
                    (color[1] + tint) & 0xFF,
                    (color[2] + tint) & 0xFF,
                )
    return img


def gen_gradient(rng: Rng, w: int, h: int) -> Image.Image:
    c0 = (rng.channel(), rng.channel(), rng.channel())
    c1 = (rng.channel(), rng.channel(), rng.channel())
    angle = rng.uniform(0.0, math.pi)
    dx, dy = math.cos(angle), math.sin(angle)
    span = max(1.0, abs(dx) * (w - 1) + abs(dy) * (h - 1))
    mode = rng.integer(0, 2)  # linear, radial, multi-stop
    img = Image.new("RGB", (w, h))
    px = img.load()
    cx, cy = w / 2.0, h / 2.0
    rmax = math.hypot(cx, cy)
    for y in range(h):
        for x in range(w):
            if mode == 1:
                t = math.hypot(x - cx, y - cy) / rmax
            else:
                t = ((x * dx + y * dy) + span / 2.0) / span
            t = max(0.0, min(1.0, t))
            if mode == 2 and t > 0.5:
                t2 = (t - 0.5) * 2.0
                px[x, y] = (
                    int(c1[0] + (c0[0] - c1[0]) * t2),
                    int(c1[1] + (c0[1] - c1[1]) * t2),
                    int(c1[2] + (c0[2] - c1[2]) * t2),
                )
            else:
                px[x, y] = (
                    int(c0[0] + (c1[0] - c0[0]) * t),
                    int(c0[1] + (c1[1] - c0[1]) * t),
                    int(c0[2] + (c1[2] - c0[2]) * t),
                )
    return img


def gen_hardedge(rng: Rng, w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h))
    px = img.load()
    palette = [
        (rng.channel(), rng.channel(), rng.channel())
        for _ in range(rng.integer(2, 6))
    ]
    kind = rng.integer(0, 3)
    # Sharp boundaries: rectangles, checker cells, diagonal bands, vertical bars
    cell = rng.integer(4, 16)
    for y in range(h):
        for x in range(w):
            if kind == 0:
                idx = (x // cell + y // cell) % len(palette)
            elif kind == 1:
                idx = ((x // cell) * (y // cell)) % len(palette)
            elif kind == 2:
                idx = ((x + y) // cell) % len(palette)
            else:
                idx = (x // cell) % len(palette)
            px[x, y] = palette[idx]
    return img


def gen_noise(rng: Rng, w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h))
    px = img.load()
    amplitude = rng.integer(16, 255)
    base = (rng.channel(), rng.channel(), rng.channel())
    kind = rng.integer(0, 2)  # white, correlated, block quantized
    block = rng.integer(2, 8)
    prev = [base[0], base[1], base[2]]
    for y in range(h):
        for x in range(w):
            if kind == 0:
                px[x, y] = (rng.channel(), rng.channel(), rng.channel())
            elif kind == 1:
                n = rng.integer(-amplitude // 2, amplitude // 2)
                prev = [
                    max(0, min(255, prev[0] + n)),
                    max(0, min(255, prev[1] + n // 2)),
                    max(0, min(255, prev[2] - n // 2)),
                ]
                px[x, y] = (prev[0], prev[1], prev[2])
            else:
                levels = rng.integer(2, 8)
                v = (rng.channel() // (256 // levels)) * (255 // max(1, levels - 1))
                v = min(255, v)
                if (x % block == 0) or (y % block == 0):
                    px[x, y] = (v, v, v)
                else:
                    left = px[x - 1, y] if x else (v, v, v)
                    px[x, y] = left
    return img


def gen_alpha(rng: Rng, w: int, h: int) -> Image.Image:
    img = Image.new("RGBA", (w, h))
    px = img.load()
    base = (rng.channel(), rng.channel(), rng.channel())
    kind = rng.integer(0, 3)  # gradient alpha, cutout, dithered, checker holes
    threshold = rng.integer(40, 220)
    cell = rng.integer(4, 24)
    for y in range(h):
        for x in range(w):
            if kind == 0:
                a = int(255 * (x + y) / max(1, w + h - 2))
            elif kind == 1:
                a = 255 if (x // cell + y // cell) % 2 == 0 else 0
            elif kind == 2:
                a = 255 if rng.integer(0, 255) > threshold else rng.integer(0, 128)
            else:
                a = 0 if (x // cell) * (y // cell) % 3 == 0 else 255
            jitter = rng.integer(-8, 8)
            px[x, y] = (
                max(0, min(255, base[0] + jitter)),
                max(0, min(255, base[1] + jitter)),
                max(0, min(255, base[2] + jitter)),
                max(0, min(255, a)),
            )
    return img


def gen_normal(rng: Rng, w: int, h: int) -> Image.Image:
    # Smooth height field -> unit normals, stored as RGB XYZ in [0,255]
    img = Image.new("RGB", (w, h))
    px = img.load()
    bumps = [
        (rng.uniform(0, w), rng.uniform(0, h), rng.uniform(4, max(5, min(w, h) / 2.0)),
         rng.uniform(0.4, 1.0))
        for _ in range(rng.integer(2, 6))
    ]
    strength = rng.uniform(1.0, 4.0)

    def height(x: float, y: float) -> float:
        z = 0.0
        for bx, by, br, amp in bumps:
            d2 = (x - bx) ** 2 + (y - by) ** 2
            z += amp * math.exp(-d2 / (2.0 * br * br))
        return z

    heights = [[height(x, y) for x in range(w)] for y in range(h)]
    for y in range(h):
        for x in range(w):
            hl = heights[y][max(0, x - 1)]
            hr = heights[y][min(w - 1, x + 1)]
            hu = heights[max(0, y - 1)][x]
            hd = heights[min(h - 1, y + 1)][x]
            nx = (hl - hr) * strength
            ny = (hu - hd) * strength
            nz = 1.0
            inv = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
            nx, ny, nz = nx * inv, ny * inv, nz * inv
            px[x, y] = (
                int(round((nx * 0.5 + 0.5) * 255)),
                int(round((ny * 0.5 + 0.5) * 255)),
                int(round((nz * 0.5 + 0.5) * 255)),
            )
    return img


GENERATORS = {
    "solid": gen_solid,
    "gradient": gen_gradient,
    "hardedge": gen_hardedge,
    "noise": gen_noise,
    "alpha": gen_alpha,
    "normal": gen_normal,
}


def main(out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    serial = 0
    for cls in CLASSES:
        cls_dir = out_root / cls
        cls_dir.mkdir(exist_ok=True)
        for i in range(COUNTS[cls]):
            rng = Rng(0xC0FFEE0000000000 ^ (serial * 0x9E3779B97F4A7C15))
            w, h = size_for(serial)
            img = GENERATORS[cls](rng, w, h)
            name = f"{cls}_{i:03d}_{w}x{h}.png"
            path = cls_dir / name
            img.save(path, format="PNG", optimize=False, compress_level=1)
            manifest_rows.append((serial, cls, name, w, h, img.mode))
            serial += 1
    assert serial == 500, serial
    with (out_root / "manifest.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(("id", "class", "file", "width", "height", "mode"))
        writer.writerows(manifest_rows)
    print(f"Generated {serial} textures under {out_root}")


if __name__ == "__main__":
    target = Path(os.environ.get("CORPUS_DIR", Path(__file__).parent / "corpus"))
    main(target)
