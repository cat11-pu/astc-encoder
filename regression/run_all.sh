#!/usr/bin/env bash
# End-to-end regression: corpus generation, two encoded runs, analysis,
# baseline comparison, and worst-sample contact sheet.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

BINARY="${ASTCENC_BINARY:-$HERE/../build-native/Source/astcenc-native}"
if [[ ! -x "$BINARY" ]]; then
    echo "Building astcenc native binary..."
    cmake -B "$HERE/../build-native" -DCMAKE_BUILD_TYPE=Release \
        -DASTCENC_ISA_NATIVE=ON -DASTCENC_UNITTEST=OFF \
        -S "$HERE/.." >/dev/null
    cmake --build "$HERE/../build-native" -j"$(nproc)"
fi

python3 gen_textures.py
python3 run_regression.py --binary "$BINARY" --label run1 --out results/run1
python3 run_regression.py --binary "$BINARY" --label run2 --out results/run2
python3 analyze.py
python3 check_baseline.py

echo "Regression complete. See results/analysis.json and REPORT.md"
