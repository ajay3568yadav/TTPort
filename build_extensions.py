#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
TTPort/build_extensions.py
==========================
Compile the DREAMPlace C++ extensions that TTPort's benchmarks use as CPU
references.  All source files are included inside this TTPort folder under
dreamplace_ref/dreamplace/ops/*/src/ — no external DREAMPlace install needed.

Built extensions are placed directly into the matching dreamplace_ref/
sub-packages so subsequent imports work automatically.

Usage:
    python build_extensions.py           # build all extensions
    python build_extensions.py --dry-run # show what would be built, don't compile

Requirements: a C++ compiler (g++ ≥ 7) and a PyTorch install that includes
torch/extension.h (standard pip install covers this).
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil

# ── Locate key directories ─────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "dreamplace_ref")
OPS_DIR = os.path.join(REF_DIR, "dreamplace", "ops")

# Make sure we can import torch even before any sys.path surgery
try:
    import torch
    from torch.utils.cpp_extension import load as cpp_load
except ImportError:
    print("ERROR: PyTorch not found. Install it with: pip install torch")
    sys.exit(1)

# ── Shared utility sources (linked into every extension) ──────────────────────
UTIL_SRC = os.path.join(OPS_DIR, "utility", "src")
UTILITY_SOURCES = [os.path.join(UTIL_SRC, "msg.cpp")]

# Extra include dirs shared by all extensions
COMMON_INCLUDES = [
    OPS_DIR,            # allows  #include "utility/src/..."  style includes
    UTIL_SRC,
]

# OpenMP flags (adds significant speed to density scatter + wirelength)
EXTRA_CFLAGS = ["-O3", "-fopenmp"]
EXTRA_LDFLAGS = ["-fopenmp"]


def _build(name: str, sources: list[str], extra_includes: list[str],
           dest_dir: str, dry_run: bool) -> bool:
    """Build one extension and copy the .so into dest_dir."""
    all_sources   = UTILITY_SOURCES + sources
    all_includes  = COMMON_INCLUDES + extra_includes

    print(f"\n  Building {name} ...")
    for s in sources:
        print(f"    src: {os.path.relpath(s, HERE)}")

    if dry_run:
        print(f"    --> would be placed in: {os.path.relpath(dest_dir, HERE)}")
        return True

    try:
        ext = cpp_load(
            name           = name,
            sources        = all_sources,
            extra_include_paths = all_includes,
            extra_cflags   = EXTRA_CFLAGS,
            extra_ldflags  = EXTRA_LDFLAGS,
            verbose        = False,
        )
    except Exception as e:
        print(f"    ERROR: {e}")
        return False

    # torch.utils.cpp_extension.load() returns the module; its .so lives in
    # torch's JIT cache.  Find it and copy to dest_dir.
    so_path = getattr(ext, "__file__", None)
    if not so_path or not os.path.isfile(so_path):
        print(f"    ERROR: Could not locate built .so for {name}")
        return False

    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(so_path))
    shutil.copy2(so_path, dest)
    print(f"    --> {os.path.relpath(dest, HERE)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be built without compiling")
    args = parser.parse_args()

    dry = args.dry_run
    print("=== TTPort build_extensions.py ===")
    if dry:
        print("  (dry-run — nothing will be compiled)")

    # ── 1. electric_potential_cpp ─────────────────────────────────────────────
    EP_SRC = os.path.join(OPS_DIR, "electric_potential", "src")
    ok1 = _build(
        name    = "electric_potential_cpp",
        sources = [
            os.path.join(EP_SRC, "electric_density_map.cpp"),
            os.path.join(EP_SRC, "electric_force.cpp"),
        ],
        extra_includes = [EP_SRC],
        dest_dir = os.path.join(OPS_DIR, "electric_potential"),
        dry_run  = dry,
    )

    # ── 2. dct2_fft2_cpp ─────────────────────────────────────────────────────
    DCT_SRC = os.path.join(OPS_DIR, "dct", "src")
    ok2 = _build(
        name    = "dct2_fft2_cpp",
        sources = [os.path.join(DCT_SRC, "dct2_fft2.cpp")],
        extra_includes = [DCT_SRC],
        dest_dir = os.path.join(OPS_DIR, "dct"),
        dry_run  = dry,
    )

    # ── 3. dct_cpp ───────────────────────────────────────────────────────────
    ok3 = _build(
        name    = "dct_cpp",
        sources = [
            os.path.join(DCT_SRC, "dct.cpp"),
            os.path.join(DCT_SRC, "dst.cpp"),
            os.path.join(DCT_SRC, "dxt.cpp"),
            os.path.join(DCT_SRC, "dct_2N.cpp"),
        ],
        extra_includes = [DCT_SRC],
        dest_dir = os.path.join(OPS_DIR, "dct"),
        dry_run  = dry,
    )

    # ── 4. dct_lee_cpp ───────────────────────────────────────────────────────
    ok4 = _build(
        name    = "dct_lee_cpp",
        sources = [os.path.join(DCT_SRC, "dct_lee.cpp")],
        extra_includes = [DCT_SRC],
        dest_dir = os.path.join(OPS_DIR, "dct"),
        dry_run  = dry,
    )

    # ── 5. weighted_average_wirelength_cpp_merged ────────────────────────────
    WA_SRC = os.path.join(OPS_DIR, "weighted_average_wirelength", "src")
    ok5 = _build(
        name    = "weighted_average_wirelength_cpp_merged",
        sources = [os.path.join(WA_SRC, "weighted_average_wirelength_merged.cpp")],
        extra_includes = [WA_SRC],
        dest_dir = os.path.join(OPS_DIR, "weighted_average_wirelength"),
        dry_run  = dry,
    )

    # ── 6. logsumexp_wirelength_cpp_merged ───────────────────────────────────
    LSE_SRC = os.path.join(OPS_DIR, "logsumexp_wirelength", "src")
    ok6 = _build(
        name    = "logsumexp_wirelength_cpp_merged",
        sources = [os.path.join(LSE_SRC, "logsumexp_wirelength_merged.cpp")],
        extra_includes = [LSE_SRC],
        dest_dir = os.path.join(OPS_DIR, "logsumexp_wirelength"),
        dry_run  = dry,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    results = {
        "electric_potential_cpp":              ok1,
        "dct2_fft2_cpp":                       ok2,
        "dct_cpp":                             ok3,
        "dct_lee_cpp":                         ok4,
        "weighted_average_wirelength_cpp_merged": ok5,
        "logsumexp_wirelength_cpp_merged":     ok6,
    }
    all_ok = all(results.values())
    for name, ok in results.items():
        tag = "  [OK]" if ok else "  [!!]"
        print(f"{tag}  {name}")

    print()
    if dry:
        print("Dry-run complete. Re-run without --dry-run to compile.")
    elif all_ok:
        print("All extensions built. TTPort is now self-contained.")
        print("Run 'bash setup.sh' to verify imports.")
    else:
        print("Some extensions failed to build. Check errors above.")
        print("Common fixes:")
        print("  - Install a C++ compiler: sudo apt install build-essential")
        print("  - Install PyTorch dev: pip install torch")
        sys.exit(1)


if __name__ == "__main__":
    main()
