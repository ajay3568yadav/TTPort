#!/usr/bin/env bash
# =============================================================================
# TTPort/setup.sh  —  Provision compiled C++ extensions inside dreamplace_ref
# =============================================================================
#
# This script makes TTPort fully self-contained.  It copies compiled .so
# extension files into dreamplace_ref/ so every benchmark runs without any
# dependency on files outside this TTPort folder.
#
# Two modes:
#   1. Pre-built copy (fast, ~2 s):
#      Looks for already-compiled .so files in a sibling DREAMPlace tree
#      (TTPlace/DREAMPlace/), then COPIES them (not symlinks) into TTPort.
#
#   2. Build from source (slower, requires compiler + PyTorch dev headers):
#      Uses build_extensions.py to compile the C++ sources included in
#      dreamplace_ref/ directly.  No external DREAMPlace required.
#
# Usage:
#   cd TTPort
#   bash setup.sh                 # auto-detect sibling build, fall back to source
#   bash setup.sh --source-only   # always build from the included C++ sources
#
# After running, every benchmark works standalone:
#   cd benchmarks/density_scatter
#   python profile_density_scatter_cpu_vs_ttnn.py
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTPLACE_DIR="$(dirname "$SCRIPT_DIR")"
DP_DEFAULT="${TTPLACE_DIR}/DREAMPlace"
REF_DIR="${SCRIPT_DIR}/dreamplace_ref"

# ── Argument parsing ──────────────────────────────────────────────────────────
SOURCE_ONLY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-only) SOURCE_ONLY=true; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "=== TTPort setup ==="
echo "  TTPort root    : ${SCRIPT_DIR}"
echo "  dreamplace_ref : ${REF_DIR}"
echo

# ── Helper: copy extensions from a pre-built DREAMPlace tree ─────────────────
copy_from_prebuilt() {
    local DP_PKG="$1"
    echo "  Source: ${DP_PKG}"
    echo

    # Map: "source_op_subdir:dest_op_subdir:glob_pattern"
    declare -a EXT_MAP=(
        "electric_potential:electric_potential:electric_potential_cpp*.so"
        "dct:dct:dct2_fft2_cpp*.so"
        "dct:dct:dct_cpp*.so"
        "dct:dct:dct_lee_cpp*.so"
        "weighted_average_wirelength:weighted_average_wirelength:weighted_average_wirelength_cpp_merged*.so"
        "weighted_average_wirelength:weighted_average_wirelength:weighted_average_wirelength_cpp*.so"
        "weighted_average_wirelength:weighted_average_wirelength:weighted_average_wirelength_cpp_atomic*.so"
        "logsumexp_wirelength:logsumexp_wirelength:logsumexp_wirelength_cpp_merged*.so"
    )

    local COPIED=0 SKIPPED=0
    for entry in "${EXT_MAP[@]}"; do
        local SRC_SUBDIR="${entry%%:*}"
        local rest="${entry#*:}"
        local DST_SUBDIR="${rest%%:*}"
        local GLOB="${rest#*:}"

        local SRC_DIR="${DP_PKG}/ops/${SRC_SUBDIR}"
        local DST_DIR="${REF_DIR}/dreamplace/ops/${DST_SUBDIR}"

        mkdir -p "${DST_DIR}"
        touch "${DST_DIR}/__init__.py"

        for so_file in "${SRC_DIR}/"${GLOB}; do
            [[ -e "${so_file}" ]] || continue
            local fname dest
            fname="$(basename "${so_file}")"
            dest="${DST_DIR}/${fname}"

            if [[ -f "${dest}" ]]; then
                echo "  [skip]  ${DST_SUBDIR}/${fname}  (already present)"
                (( SKIPPED++ )) || true
            else
                # Remove any stale symlink before copying
                [[ -L "${dest}" ]] && rm -f "${dest}"
                cp "${so_file}" "${dest}"
                echo "  [copy]  ${DST_SUBDIR}/${fname}"
                (( COPIED++ )) || true
            fi
        done
    done

    # Also copy Python wrappers for WA / LSE ops
    for op in weighted_average_wirelength logsumexp_wirelength; do
        local op_src="${DP_PKG}/ops/${op}"
        local op_dst="${REF_DIR}/dreamplace/ops/${op}"
        mkdir -p "${op_dst}"; touch "${op_dst}/__init__.py"
        for pyf in "${op_src}"/*.py; do
            [[ -f "${pyf}" ]] || continue
            local bname; bname="$(basename "${pyf}")"
            if [[ ! -f "${op_dst}/${bname}" ]]; then
                cp "${pyf}" "${op_dst}/${bname}"
                echo "  [copy py] ops/${op}/${bname}"
            fi
        done
    done

    echo
    echo "  Done: ${COPIED} extensions copied, ${SKIPPED} already present."
}

# ── Mode selection ────────────────────────────────────────────────────────────
if [[ "${SOURCE_ONLY}" == true ]]; then
    echo ">>> Building from included C++ sources (--source-only) ..."
    python3 "${SCRIPT_DIR}/build_extensions.py"
else
    DP_PKG="${DP_DEFAULT}/dreamplace"
    if ls "${DP_PKG}/ops/electric_potential/"*_cpp*.so >/dev/null 2>&1; then
        echo ">>> Pre-built extensions found — copying into TTPort ..."
        copy_from_prebuilt "${DP_PKG}"
    else
        echo ">>> No pre-built extensions found at ${DP_PKG}."
        echo "    Falling back to building from included C++ sources ..."
        echo "    (This may take a few minutes on first run.)"
        echo
        python3 "${SCRIPT_DIR}/build_extensions.py"
    fi
fi

echo
# ── Verify imports ────────────────────────────────────────────────────────────
echo "=== Verifying imports ==="
python3 - <<'PYEOF'
import sys, os

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "dreamplace_ref")
if ref not in sys.path:
    sys.path.insert(0, ref)

results = {}

def try_import(label, mod):
    try:
        __import__(mod)
        results[label] = "OK"
    except Exception as e:
        results[label] = f"MISSING  ({e})"

try_import("electric_potential_cpp",             "dreamplace.ops.electric_potential.electric_potential_cpp")
try_import("dct2_fft2_cpp",                      "dreamplace.ops.dct.dct2_fft2_cpp")
try_import("ElectricDensityMapFunction",         "dreamplace.ops.electric_potential.electric_overflow")
try_import("TTNNFieldSolver",                    "dreamplace.ops.electric_potential.ttnn_poisson_solver")
try_import("ttnn_density_map_scatter",           "dreamplace.ops.electric_potential.ttnn_density_map_scatter")
try_import("discrete_spectral_transform",        "dreamplace.ops.dct.discrete_spectral_transform")

max_w = max(len(k) for k in results)
for k, v in results.items():
    tag = "  [OK]" if v == "OK" else "  [!!]"
    print(f"{tag}  {k:<{max_w}}  {v}")

ok = all(v == "OK" for v in results.values())
if not ok:
    print("\n  Some imports failed — run 'bash setup.sh --source-only' to build from source.")
    sys.exit(1)
PYEOF

echo
echo "=== Setup complete — TTPort is self-contained ==="
echo "Run benchmarks from their directories, e.g.:"
echo "  cd benchmarks/density_scatter"
echo "  python profile_density_scatter_cpu_vs_ttnn.py"
echo "  python profile_density_scatter_cpu_vs_ttnn.py --run-v6-kernel"
