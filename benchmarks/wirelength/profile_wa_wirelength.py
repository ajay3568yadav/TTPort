#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
WA wirelength profiling entrypoint inside TTPort.

Runs the standalone benchmark in this same directory:
  wa_wirelength_benchmark.py

That script profiles CPU (PyTorch + DREAMPlace compiled ops when available) vs TTNN
on device. No path outside TTPort is required.

Usage:
  cd TTPort/benchmarks/wirelength
  python profile_wa_wirelength.py
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(here, "wa_wirelength_benchmark.py")
    if not os.path.isfile(target):
        raise RuntimeError(f"Missing standalone benchmark: {target}")

    cmd = [sys.executable, target] + sys.argv[1:]
    print("Launching WA wirelength benchmark (TTPort-only):")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
