#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Standalone Poisson / DCT field solver benchmark: DREAMPlace CPU DCT (FFT/C++) vs TTNNFieldSolver.

Self-contained under TTPlace/dreamplace_ttnn_profile/scripts — uses DREAMPlace code only from
TTPlace/DREAMPlace (via PYTHONPATH).

**CPU reference** is the same path as ``ElectricPotentialFunction.forward`` when
``ttnn_field_solver is None``: ``dct2_fft2.DCT2`` → spectral weights → ``IDXST_IDCT`` /
``IDCT_IDXST`` (``dreamplace.ops.dct.dct2_fft2``). Requires the built ``dct2_fft2_cpp`` extension.

An optional dense **matmul** spectral solve (same formulation as ``TTNNFieldSolver``) can be
printed for sanity via ``--compare-matmul-reference``.

Usage:
  export PYTHONPATH=/path/to/TTPlace/DREAMPlace:$PYTHONPATH
  cd TTPlace/dreamplace_ttnn_profile/scripts
  python poisson_solver_cpu_vs_ttnn_benchmark.py --num-bins-x 512 --num-bins-y 512
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np
import torch

# TTPort/dreamplace_ref on path (benchmarks/field_solver/ -> benchmarks/ -> TTPort/)
_here = os.path.dirname(os.path.abspath(__file__))
_ttport = os.path.abspath(os.path.join(_here, "..", ".."))
_dp_root = os.path.join(_ttport, "dreamplace_ref")
if os.path.isdir(_dp_root) and _dp_root not in sys.path:
    sys.path.insert(0, _dp_root)

from dreamplace.ops.electric_potential.ttnn_poisson_solver import (  # noqa: E402
    HAS_TTNN,
    TTNNFieldSolver,
    _build_dct2_matrix,
    _build_idct_matrix,
    _build_idxst_matrix,
)

HAS_DREAMPLACE_DCT = False
_DREAMPLACE_DCT_IMPORT_ERROR = ""
dp_dct_mod = None
precompute_expk = None
try:
    from dreamplace.ops.dct import dct2_fft2 as dp_dct_mod  # noqa: E402
    from dreamplace.ops.dct.discrete_spectral_transform import (  # noqa: E402
        get_exact_expk as precompute_expk,
    )

    HAS_DREAMPLACE_DCT = True
except Exception as _e:
    _DREAMPLACE_DCT_IMPORT_ERROR = repr(_e)


class DreamplaceFftFieldSolver:
    """DREAMPlace electric-potential spectral field (same as CPU path in ``electric_potential``)."""

    def __init__(
        self,
        M: int,
        N: int,
        bin_size_x: float,
        bin_size_y: float,
        *,
        dtype=torch.float32,
        device=None,
    ):
        if not HAS_DREAMPLACE_DCT:
            raise RuntimeError("DREAMPlace dct2_fft2 not available")
        if device is None:
            device = torch.device("cpu")
        expkM = precompute_expk(M, dtype=dtype, device=device)
        expkN = precompute_expk(N, dtype=dtype, device=device)
        self.dct2 = dp_dct_mod.DCT2(expkM, expkN)
        self.idxst_idct = dp_dct_mod.IDXST_IDCT(expkM, expkN)
        self.idct_idxst = dp_dct_mod.IDCT_IDXST(expkM, expkN)
        wu = torch.arange(M, dtype=dtype, device=device).mul(2 * np.pi / M).view(M, 1)
        wv = (
            torch.arange(N, dtype=dtype, device=device)
            .mul(2 * np.pi / N)
            .view(1, N)
            .mul_(bin_size_x / bin_size_y)
        )
        wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
        wu2_plus_wv2[0, 0] = 1.0
        inv = 1.0 / wu2_plus_wv2
        inv[0, 0] = 0.0
        self.wu_by = wu * inv * (1.0 / 2)
        self.wv_by = wv * inv * (1.0 / 2)

    def solve(self, density_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rho = density_map.contiguous()
        auv = self.dct2(rho)
        # .clone(): DCT modules reuse internal output buffers between calls.
        fx = self.idxst_idct(auv * self.wu_by).clone()
        fy = self.idct_idxst(auv * self.wv_by).clone()
        return fx, fy


def _spectral_weights_torch(M: int, N: int, bin_size_x: float, bin_size_y: float):
    """Same wu/wv weight grids as TTNNFieldSolver (float32)."""
    wu = torch.arange(M, dtype=torch.float32).mul(2 * np.pi / M).view(M, 1)
    wv = torch.arange(N, dtype=torch.float32).mul(2 * np.pi / N).view(1, N)
    wv = wv * (bin_size_x / bin_size_y)
    wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
    wu2_plus_wv2[0, 0] = 1.0
    inv = 1.0 / wu2_plus_wv2
    inv[0, 0] = 0.0
    wu_weights = wu * inv * 0.5
    wv_weights = wv * inv * 0.5
    return wu_weights, wv_weights


def cpu_matmul_field_solve(
    density_map: torch.Tensor,
    bin_size_x: float,
    bin_size_y: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dense matmul spectral solve (same math as ``TTNNFieldSolver``), for optional sanity check.

    density_map: (M, N) float32, already normalized by 1/(bin_size_x * bin_size_y).
    """
    M, N = density_map.shape
    rho = density_map.float().contiguous()

    DCT_M = torch.tensor(_build_dct2_matrix(M), dtype=torch.float32)
    DCT_N = torch.tensor(_build_dct2_matrix(N), dtype=torch.float32)
    IDXST_M = torch.tensor(_build_idxst_matrix(M), dtype=torch.float32)
    IDCT_N = torch.tensor(_build_idct_matrix(N), dtype=torch.float32)
    IDCT_M = torch.tensor(_build_idct_matrix(M), dtype=torch.float32)
    IDXST_N = torch.tensor(_build_idxst_matrix(N), dtype=torch.float32)

    auv = DCT_M @ rho @ DCT_N.T
    wu_w, wv_w = _spectral_weights_torch(M, N, bin_size_x, bin_size_y)
    fx_auv = auv * wu_w
    fy_auv = auv * wv_w

    field_x = 2.0 * (IDXST_M @ fx_auv @ IDCT_N.T)
    field_y = 2.0 * (IDCT_M @ fy_auv @ IDXST_N.T)
    return field_x.contiguous(), field_y.contiguous()


def _accuracy(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a.float() - b.float()).reshape(-1)
    ref = b.float().reshape(-1)
    abs_d = d.abs()
    return {
        "max_abs": float(abs_d.max()),
        "mean_abs": float(abs_d.mean()),
        "rmse": float(torch.sqrt((d * d).mean())),
        "rel_l2": float(d.norm(p=2) / (ref.norm(p=2) + 1e-12)),
    }


def _make_density(M: int, N: int, seed: int, die_w: float, die_h: float) -> tuple[torch.Tensor, float, float]:
    bsx = die_w / M
    bsy = die_h / N
    g = torch.Generator().manual_seed(seed)
    # Raw bin occupancies (unnormalized); then normalize like DREAMPlace after scatter.
    raw = torch.rand(M, N, generator=g, dtype=torch.float32) * 0.5 + 0.01
    normalized = raw / (bsx * bsy)
    return normalized, bsx, bsy


def main() -> None:
    p = argparse.ArgumentParser(
        description="Poisson/DCT field solver: DREAMPlace FFT DCT (CPU) vs TTNNFieldSolver"
    )
    p.add_argument("--num-bins-x", type=int, default=512)
    p.add_argument("--num-bins-y", type=int, default=512)
    p.add_argument("--die-width", type=float, default=12000.0)
    p.add_argument("--die-height", type=float, default=12000.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--skip-ttnn", action="store_true", help="CPU only (no TT device)")
    p.add_argument(
        "--compare-matmul-reference",
        action="store_true",
        help="Report dense matmul spectral solve vs DREAMPlace DCT (one-shot sanity)",
    )
    args = p.parse_args()

    if not HAS_DREAMPLACE_DCT:
        print("ERROR: DREAMPlace DCT operators not available (need built dct2_fft2 extension).")
        print(f"  Import error: {_DREAMPLACE_DCT_IMPORT_ERROR}")
        print("  Build DREAMPlace with dct2_fft2_cpp, or use a checkout where it imports.")
        sys.exit(1)

    M, N = args.num_bins_x, args.num_bins_y
    density, bsx, bsy = _make_density(M, N, args.seed, args.die_width, args.die_height)

    print("=" * 72)
    print("Poisson / DCT: DREAMPlace CPU DCT (dct2_fft2) vs TTNNFieldSolver")
    print("=" * 72)
    print(f"  Grid     : {M} x {N}")
    print(f"  Die      : {args.die_width} x {args.die_height}")
    print(f"  Bin      : {bsx:.4f} x {bsy:.4f}")
    print(f"  Warmup   : {args.warmup}   Iters: {args.iters}")
    print(f"  HAS_TTNN : {HAS_TTNN}")
    print("  DREAMPlace DCT (dct2_fft2): OK")
    print()

    dp_solver = DreamplaceFftFieldSolver(M, N, bsx, bsy, dtype=torch.float32, device=density.device)

    if args.compare_matmul_reference:
        fx_mm, fy_mm = cpu_matmul_field_solve(density, bsx, bsy)
        fx_dp0, fy_dp0 = dp_solver.solve(density)
        mx = _accuracy(fx_mm, fx_dp0)
        my = _accuracy(fy_mm, fy_dp0)
        print("[0] Dense matmul vs DREAMPlace DCT (sanity, one shot) ...")
        print(f"  field_x  rel_l2={mx['rel_l2']:.4e}  max_abs={mx['max_abs']:.4e}")
        print(f"  field_y  rel_l2={my['rel_l2']:.4e}  max_abs={my['max_abs']:.4e}")
        print()

    # --- DREAMPlace CPU DCT ---
    print("[1/2] DREAMPlace CPU DCT (DCT2 → IDXST_IDCT / IDCT_IDXST) ...")
    cpu_times: list[float] = []
    fx_cpu = fy_cpu = None
    for i in range(args.warmup + args.iters):
        t0 = time.perf_counter()
        fx_cpu, fy_cpu = dp_solver.solve(density)
        t_ms = (time.perf_counter() - t0) * 1e3
        if i >= args.warmup:
            cpu_times.append(t_ms)
    print(f"  mean={statistics.mean(cpu_times):9.3f} ms  "
          f"p50={statistics.median(cpu_times):9.3f}  "
          f"min={min(cpu_times):9.3f}  max={max(cpu_times):9.3f}")
    print()

    if args.skip_ttnn or not HAS_TTNN:
        print("[2/2] TTNNFieldSolver — skipped (--skip-ttnn or ttnn unavailable)")
        print("\nDone.")
        return

    # --- TTNN ---
    print("[2/2] TTNNFieldSolver.solve() (upload + compute + download) ...")
    import ttnn

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=8192)
    try:
        solver = TTNNFieldSolver(M, N, bsx, bsy, device=device, device_id=args.device_id)
        # Warmup: first solve(s) may compile kernels; do not include in reported averages.
        for _ in range(args.warmup):
            solver.solve(density.clone())
            ttnn.synchronize_device(device)
        solver.reset_timing_stats()

        tt_times: list[float] = []
        fx_tt = fy_tt = None
        for _ in range(args.iters):
            rho = density.clone()
            t0 = time.perf_counter()
            fx_tt, fy_tt = solver.solve(rho)
            ttnn.synchronize_device(device)
            tt_times.append((time.perf_counter() - t0) * 1e3)
        print(f"  mean={statistics.mean(tt_times):9.3f} ms  "
              f"p50={statistics.median(tt_times):9.3f}  "
              f"min={min(tt_times):9.3f}  max={max(tt_times):9.3f}")
        # Matches steady-state iters only (same window as mean wall time above).
        print(f"  {solver.report_timing()}  (steady-state, post-warmup)")

        acc_x = _accuracy(fx_tt, fx_cpu)
        acc_y = _accuracy(fy_tt, fy_cpu)
        print()
        print("  vs DREAMPlace CPU DCT: field_x  "
              f"max_abs={acc_x['max_abs']:.4e}  mean_abs={acc_x['mean_abs']:.4e}  "
              f"rmse={acc_x['rmse']:.4e}  rel_l2={acc_x['rel_l2']:.4e}")
        print("  vs DREAMPlace CPU DCT: field_y  "
              f"max_abs={acc_y['max_abs']:.4e}  mean_abs={acc_y['mean_abs']:.4e}  "
              f"rmse={acc_y['rmse']:.4e}  rel_l2={acc_y['rel_l2']:.4e}")

        cpu_m = statistics.mean(cpu_times)
        tt_m = statistics.mean(tt_times)
        print()
        print(f"  Speedup (DREAMPlace CPU DCT / TTNN wall): {cpu_m / max(tt_m, 1e-9):.3f}x")
        solver.close()
    finally:
        ttnn.close_device(device)

    print("\nDone.")


if __name__ == "__main__":
    main()
