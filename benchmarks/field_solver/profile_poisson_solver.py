#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Poisson / DCT field solver profiling entrypoint inside TTPort.

Runs the standalone benchmark in this same directory:
  poisson_solver_benchmark.py

CPU reference uses the same matmul spectral pipeline as ``TTNNFieldSolver``.
Extra CLI arguments are forwarded (e.g. --num-bins-x 512 --skip-ttnn).

Usage:
  cd TTPort/benchmarks/field_solver
  python profile_poisson_solver.py
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(here, "poisson_solver_benchmark.py")
    if not os.path.isfile(target):
        raise RuntimeError(f"Missing standalone benchmark: {target}")

    cmd = [sys.executable, target] + sys.argv[1:]
    print("Launching Poisson/DCT benchmark (TTPort-only):")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
