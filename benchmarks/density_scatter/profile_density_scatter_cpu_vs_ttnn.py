#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark density-map scatter: DREAMPlace CPU path vs TTNN scatter paths.

Three paths are compared:
  1. CPU C++   — ElectricDensityMapFunction.forward (compiled DREAMPlace kernel,
                  uses triangle_density_function from density_function.h)
  2. CPU PyTorch — pure-PyTorch reference using the exact same triangle formula
                   as the C++ kernel (min/max overlap, clipped to 0), for
                   diagnostic comparison.
  3. TTNN orig  — density_map_scatter_ttnn (overlap computed on-device via
                   abs+clip, aggregated with chunked matmul)
  4. TTNN accur — density_map_scatter_ttnn_accurate (overlap on CPU using exact
                   C++ formula, aggregated with chunked TTNN matmul)

Runtime and accuracy (vs CPU C++) are reported for all paths.

Usage:
  export PYTHONPATH=/home/ubuntu/ayadav/TT_Port/TTPlace/DREAMPlace:$PYTHONPATH
  python profile_density_map_scatter_cpu_vs_ttnn.py --num-bins-x 512 --num-bins-y 512
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

def _setup_pythonpath() -> None:
    # benchmarks/density_scatter/ -> benchmarks/ -> TTPort/ -> dreamplace_ref/
    dreamplace_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "dreamplace_ref")
    )
    dreamplace_pkg = os.path.join(dreamplace_root, "dreamplace")
    for p in (dreamplace_root, dreamplace_pkg):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Synthetic placement input generation
# ---------------------------------------------------------------------------

@dataclass
class DensityInputs:
    # Full [2*n] pos vector used by the C++ kernel: [x_0..x_{n-1}, y_0..y_{n-1}]
    pos_all: torch.Tensor
    # Per-node arrays ordered [movable(0..m-1), fixed(m..m+t-1), filler(n-f..n-1)]
    node_size_x_clamped_all: torch.Tensor
    node_size_y_clamped_all: torch.Tensor
    offset_x_all: torch.Tensor
    offset_y_all: torch.Tensor
    ratio_all: torch.Tensor
    # Movable+filler slices for TTNN path (raw pos, offset, size, ratio)
    pos_x_mf: torch.Tensor
    pos_y_mf: torch.Tensor
    node_size_x_clamped_mf: torch.Tensor
    node_size_y_clamped_mf: torch.Tensor
    offset_x_mf: torch.Tensor
    offset_y_mf: torch.Tensor
    ratio_mf: torch.Tensor
    # Grid helpers
    bin_center_x: torch.Tensor   # (K,)  xl + (k+0.5)*bin_size_x
    bin_center_y: torch.Tensor   # (H,)
    bin_left_x: torch.Tensor     # (K,)  xl + k*bin_size_x  — matches C++ triangle formula
    bin_left_y: torch.Tensor     # (H,)
    initial_density_map: torch.Tensor
    padding_mask: torch.Tensor
    sorted_node_map: torch.Tensor
    num_movable_impacted_bins_x: int
    num_movable_impacted_bins_y: int
    num_filler_impacted_bins_x: int
    num_filler_impacted_bins_y: int


def _build_inputs(args: argparse.Namespace) -> Tuple[DensityInputs, Dict]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    xl, yl = 0.0, 0.0
    xh, yh = float(args.die_width), float(args.die_height)
    K, H = int(args.num_bins_x), int(args.num_bins_y)
    bsx = (xh - xl) / K
    bsy = (yh - yl) / H
    sqrt2 = math.sqrt(2.0)

    m = int(args.num_movable)
    t = int(args.num_fixed)
    f = int(args.num_filler)
    n = m + t + f

    size_x = torch.empty(n).uniform_(args.min_cell_w * bsx, args.max_cell_w * bsx)
    size_y = torch.empty(n).uniform_(args.min_cell_h * bsy, args.max_cell_h * bsy)

    pos_x = torch.empty(n).uniform_(xl, xh - size_x.mean())
    pos_y = torch.empty(n).uniform_(yl, yh - size_y.mean())
    # torch.clamp cannot mix scalar min with tensor max in one call.
    # Clamp in two steps to keep each cell inside [xl, xh - size_x] and [yl, yh - size_y].
    pos_x = torch.maximum(pos_x, torch.full_like(pos_x, float(xl)))
    pos_x = torch.minimum(pos_x, torch.full_like(pos_x, float(xh)) - size_x)
    pos_y = torch.maximum(pos_y, torch.full_like(pos_y, float(yl)))
    pos_y = torch.minimum(pos_y, torch.full_like(pos_y, float(yh)) - size_y)
    pos_all = torch.cat([pos_x, pos_y]).contiguous()

    sx_cl  = size_x.clamp(min=bsx * sqrt2)
    sy_cl  = size_y.clamp(min=bsy * sqrt2)
    off_x  = (size_x - sx_cl) * 0.5
    off_y  = (size_y - sy_cl) * 0.5
    ratio  = (size_x * size_y) / (sx_cl * sy_cl)

    # Movable+filler indexing — matches how C++ kernel iterates:
    #   movable: [0, m), filler: [n-f, n)
    if f > 0:
        mf_idx = torch.cat([torch.arange(m), torch.arange(n - f, n)])
    else:
        mf_idx = torch.arange(m)

    bin_left_x  = torch.tensor(xl + np.arange(K, dtype=np.float32) * bsx, dtype=torch.float32)
    bin_left_y  = torch.tensor(yl + np.arange(H, dtype=np.float32) * bsy, dtype=torch.float32)
    bin_center_x = bin_left_x + float(bsx) / 2
    bin_center_y = bin_left_y + float(bsy) / 2

    def _impacted(arr, max_bins, bin_size):
        if len(arr) == 0:
            return 0
        return int(
            ((arr.max() + 2 * sqrt2 * bin_size) / bin_size)
            .ceil().clamp(max=max_bins).item()
        )

    data = DensityInputs(
        pos_all                     = pos_all,
        node_size_x_clamped_all    = sx_cl.contiguous(),
        node_size_y_clamped_all    = sy_cl.contiguous(),
        offset_x_all               = off_x.contiguous(),
        offset_y_all               = off_y.contiguous(),
        ratio_all                  = ratio.contiguous(),
        pos_x_mf                   = pos_x[mf_idx].contiguous(),
        pos_y_mf                   = pos_y[mf_idx].contiguous(),
        node_size_x_clamped_mf     = sx_cl[mf_idx].contiguous(),
        node_size_y_clamped_mf     = sy_cl[mf_idx].contiguous(),
        offset_x_mf                = off_x[mf_idx].contiguous(),
        offset_y_mf                = off_y[mf_idx].contiguous(),
        ratio_mf                   = ratio[mf_idx].contiguous(),
        bin_center_x               = bin_center_x.contiguous(),
        bin_center_y               = bin_center_y.contiguous(),
        bin_left_x                 = bin_left_x.contiguous(),
        bin_left_y                 = bin_left_y.contiguous(),
        initial_density_map        = torch.zeros(K, H, dtype=torch.float32),
        padding_mask               = torch.zeros(K, H, dtype=torch.uint8),
        sorted_node_map            = torch.arange(m, dtype=torch.int32),
        num_movable_impacted_bins_x = _impacted(sx_cl[:m], K, bsx),
        num_movable_impacted_bins_y = _impacted(sy_cl[:m], H, bsy),
        num_filler_impacted_bins_x  = _impacted(sx_cl[n - f:] if f > 0 else sx_cl[:0], K, bsx),
        num_filler_impacted_bins_y  = _impacted(sy_cl[n - f:] if f > 0 else sy_cl[:0], H, bsy),
    )
    meta = dict(xl=xl, yl=yl, xh=xh, yh=yh,
                bin_size_x=bsx, bin_size_y=bsy,
                target_density=float(args.target_density),
                num_movable=m, num_fixed=t, num_filler=f, n=n,
                num_bins_x=K, num_bins_y=H)
    return data, meta


# ---------------------------------------------------------------------------
# CPU PyTorch reference: exact triangle_density_function formula from C++
# ---------------------------------------------------------------------------

def _cpu_pytorch_reference(data: DensityInputs, meta: Dict) -> torch.Tensor:
    """
    Pure-PyTorch implementation of the DREAMPlace triangle density map scatter.

    Matches the C++ ``triangle_density_function`` in ``density_function.h``::

        overlap_x[i,k] = max(0, min(eff_x[i]+sx[i], xl+(k+1)*bsx)
                                - max(eff_x[i], xl+k*bsx))

    Processes the same movable+filler cells the C++ kernel would.
    All arithmetic in float32 on CPU, same as the C++ kernel.
    """
    m   = meta["num_movable"]
    f   = meta["num_filler"]
    n   = meta["n"]
    K   = meta["num_bins_x"]
    H   = meta["num_bins_y"]
    bsx = meta["bin_size_x"]
    bsy = meta["bin_size_y"]

    pos_all = data.pos_all
    pos_x_all = pos_all[:n]
    pos_y_all = pos_all[n:]

    if f > 0:
        mf_idx = torch.cat([torch.arange(m), torch.arange(n - f, n)])
    else:
        mf_idx = torch.arange(m)

    eff_x = (pos_x_all[mf_idx] + data.offset_x_all[mf_idx]).float().view(-1, 1)
    eff_y = (pos_y_all[mf_idx] + data.offset_y_all[mf_idx]).float().view(-1, 1)
    sx    = data.node_size_x_clamped_all[mf_idx].float().view(-1, 1)
    sy    = data.node_size_y_clamped_all[mf_idx].float().view(-1, 1)
    r     = data.ratio_all[mf_idx].float().view(-1, 1)

    bl_x = data.bin_left_x.view(1, K)   # (1, K)
    bl_y = data.bin_left_y.view(1, H)   # (1, H)

    # Exact C++ formula — clamp to 0 (C++ skips non-overlapping bins via loop bounds)
    Px = torch.clamp(
        torch.minimum(eff_x + sx, bl_x + bsx) - torch.maximum(eff_x, bl_x),
        min=0.0,
    )  # (C, K)
    Py = torch.clamp(
        torch.minimum(eff_y + sy, bl_y + bsy) - torch.maximum(eff_y, bl_y),
        min=0.0,
    )  # (C, H)

    density_map = data.initial_density_map.clone() + Px.t() @ (Py * r)
    return density_map.contiguous()


def _precompute_scatter_add_inputs(data: DensityInputs, meta: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build flattened (index, src) tensors for TTNN scatter_add from movable+filler cells.
    """
    K = meta["num_bins_x"]
    H = meta["num_bins_y"]
    bsx = float(meta["bin_size_x"])
    bsy = float(meta["bin_size_y"])
    xl = float(meta["xl"])
    yl = float(meta["yl"])

    # Effective stretched boxes, matching DREAMPlace density map kernel inputs.
    ex = (data.pos_x_mf + data.offset_x_mf).float()
    ey = (data.pos_y_mf + data.offset_y_mf).float()
    sx = data.node_size_x_clamped_mf.float()
    sy = data.node_size_y_clamped_mf.float()
    rr = data.ratio_mf.float()

    inv_bx = 1.0 / bsx
    inv_by = 1.0 / bsy
    kl = torch.floor((ex - xl) * inv_bx).to(torch.int64).clamp(min=0, max=K - 1)
    kh = (torch.floor((ex + sx - xl) * inv_bx).to(torch.int64) + 1).clamp(max=K)
    hl = torch.floor((ey - yl) * inv_by).to(torch.int64).clamp(min=0, max=H - 1)
    hh = (torch.floor((ey + sy - yl) * inv_by).to(torch.int64) + 1).clamp(max=H)

    max_kspan = int((kh - kl).max().item())
    max_hspan = int((hh - hl).max().item())

    idx_parts = []
    src_parts = []
    for dk in range(max_kspan):
        k_idx = kl + dk
        valid_k = k_idx < kh
        if not valid_k.any():
            continue
        for dh in range(max_hspan):
            h_idx = hl + dh
            valid = valid_k & (h_idx < hh)
            if not valid.any():
                continue
            k_v = k_idx[valid].float()
            h_v = h_idx[valid].float()
            ex_v = ex[valid]
            ey_v = ey[valid]
            sx_v = sx[valid]
            sy_v = sy[valid]
            rr_v = rr[valid]

            bin_xl = xl + k_v * bsx
            bin_yl = yl + h_v * bsy
            px = torch.clamp(torch.minimum(ex_v + sx_v, bin_xl + bsx) - torch.maximum(ex_v, bin_xl), min=0.0)
            py = torch.clamp(torch.minimum(ey_v + sy_v, bin_yl + bsy) - torch.maximum(ey_v, bin_yl), min=0.0)
            area = (px * py * rr_v).float()
            nz = area > 0
            if nz.any():
                flat_idx = (k_v[nz].to(torch.int64) * H + h_v[nz].to(torch.int64)).to(torch.int32)
                idx_parts.append(flat_idx)
                src_parts.append(area[nz])

    if not idx_parts:
        return torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.float32)
    return torch.cat(idx_parts), torch.cat(src_parts)


def _ttnn_scatter_add_once(
    ttnn: Any, device: Any, index_cpu: torch.Tensor, src_cpu: torch.Tensor, num_bins: int
) -> torch.Tensor:
    """Run one TTNN scatter_add and return flattened density vector on CPU."""
    k = index_cpu.numel()
    k_pad = ((k + 31) // 32) * 32
    d_pad = ((num_bins + 31) // 32) * 32

    idx2d = torch.zeros((1, k_pad), dtype=torch.int32)
    idx2d[0, :k] = index_cpu
    src2d = torch.zeros((1, k_pad), dtype=torch.float32)
    src2d[0, :k] = src_cpu
    inp2d = torch.zeros((1, d_pad), dtype=torch.float32)

    input_tt = ttnn.from_torch(inp2d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    src_tt = ttnn.from_torch(src2d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    index_tt = ttnn.from_torch(idx2d, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    result_tt = ttnn.scatter_add(input_tt, dim=1, index=index_tt, src=src_tt)
    ttnn.synchronize_device(device)
    out = ttnn.to_torch(result_tt).float()[0, :num_bins].contiguous()
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stats(samples_ms: list) -> Dict:
    return dict(
        mean=float(statistics.mean(samples_ms)),
        p50 =float(statistics.median(samples_ms)),
        min =float(min(samples_ms)),
        max =float(max(samples_ms)),
    )


def _accuracy(candidate: torch.Tensor, reference: Optional[torch.Tensor]) -> Dict:
    if reference is None:
        return dict(max_abs=float("nan"), mean_abs=float("nan"),
                    rmse=float("nan"), rel_l2=float("nan"))
    diff      = candidate.float() - reference.float()
    abs_diff  = diff.abs()
    rel_l2    = float(diff.norm(p=2) / (reference.norm(p=2) + 1e-12))
    return dict(
        max_abs  = float(abs_diff.max()),
        mean_abs = float(abs_diff.mean()),
        rmse     = float(torch.sqrt((diff * diff).mean())),
        rel_l2   = rel_l2,
    )


def _fmt_stats(s: Dict) -> str:
    return f"mean={s['mean']:9.3f}  p50={s['p50']:9.3f}  min={s['min']:9.3f}  max={s['max']:9.3f} ms"


def _fmt_acc(a: Dict) -> str:
    return (f"max_abs={a['max_abs']:.4e}  mean_abs={a['mean_abs']:.4e}"
            f"  rmse={a['rmse']:.4e}  rel_l2={a['rel_l2']:.4e}")


def _load_tt_v6_helpers():
    # benchmarks/density_scatter/ -> benchmarks/ -> TTPort/
    helper_path = pathlib.Path(__file__).resolve().parents[2] / "tt_kernels" / "v6_kernel_launcher.py"
    if not helper_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("tt_v6_helpers", str(helper_path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DREAMPlace density-map scatter: CPU C++, CPU PyTorch, TTNN."
    )
    parser.add_argument("--num-bins-x",   type=int,   default=512)
    parser.add_argument("--num-bins-y",   type=int,   default=512)
    parser.add_argument("--num-movable",  type=int,   default=200000)
    parser.add_argument("--num-fixed",    type=int,   default=10000)
    parser.add_argument("--num-filler",   type=int,   default=20000)
    parser.add_argument("--die-width",    type=float, default=12000.0)
    parser.add_argument("--die-height",   type=float, default=12000.0)
    parser.add_argument("--target-density", type=float, default=0.7)
    parser.add_argument("--min-cell-w",   type=float, default=0.2)
    parser.add_argument("--max-cell-w",   type=float, default=3.5)
    parser.add_argument("--min-cell-h",   type=float, default=0.2)
    parser.add_argument("--max-cell-h",   type=float, default=3.5)
    parser.add_argument("--chunk-size",   type=int,   default=32768)
    parser.add_argument("--warmup",       type=int,   default=2)
    parser.add_argument("--iters",        type=int,   default=5)
    parser.add_argument("--skip-ttnn-orig",   action="store_true",
                        help="Skip the original full-device TTNN path (saves time)")
    parser.add_argument("--skip-ttnn-accur",  action="store_true",
                        help="Skip the accurate CPU-overlap+TT-matmul TTNN path")
    parser.add_argument("--skip-ttnn-scatter-add", action="store_true",
                        help="Skip TTNN scatter_add benchmark path")
    parser.add_argument("--run-v6-kernel", action="store_true",
                        help="Also profile custom TT v6 kernel path")
    parser.add_argument("--seed",         type=int,   default=1234)
    args = parser.parse_args()

    _setup_pythonpath()
    # The CPU C++ reference requires the compiled electric_potential_cpp extension
    # (built with the full DREAMPlace CMake build). If not available, path [1] is skipped.
    try:
        from dreamplace.ops.electric_potential.electric_overflow import ElectricDensityMapFunction
        HAS_CPU_CPP = True
    except ImportError:
        ElectricDensityMapFunction = None
        HAS_CPU_CPP = False
        print("[INFO] electric_potential_cpp not available — CPU C++ path will be skipped.")
        print("       Build DREAMPlace with CMake and add its build dir to PYTHONPATH for that path.\n")
    from dreamplace.ops.electric_potential import ttnn_density_map_scatter as tt_mod

    data, meta = _build_inputs(args)

    print("=" * 72)
    print("Density Scatter Benchmark")
    print("=" * 72)
    print(f"  Grid   : {meta['num_bins_x']} x {meta['num_bins_y']} bins")
    print(f"  Cells  : movable={meta['num_movable']}  fixed={meta['num_fixed']}"
          f"  filler={meta['num_filler']}")
    print(f"  Die    : {meta['xh']} x {meta['yh']}")
    print(f"  Bin    : {meta['bin_size_x']:.4f} x {meta['bin_size_y']:.4f}")
    print(f"  Warmup : {args.warmup}   Iters: {args.iters}")
    print()

    # ------------------------------------------------------------------ #
    # 1. CPU C++ reference (exact DREAMPlace kernel)                      #
    # ------------------------------------------------------------------ #
    cpu_cpp_ref: Optional[torch.Tensor] = None
    cpp_stats: Optional[Dict] = None
    if not HAS_CPU_CPP:
        print("[1/4] CPU C++ — SKIPPED (electric_potential_cpp not built)")
        print("      Add DREAMPlace build dir to PYTHONPATH to enable this path.\n")
    else:
        print("[1/4] CPU C++ (ElectricDensityMapFunction — DREAMPlace kernel) ...")
        cpu_cpp_times_ms: list = []
        cpu_cpp_out: Optional[torch.Tensor] = None
        for i in range(args.warmup + args.iters):
            t0 = time.perf_counter()
            out = ElectricDensityMapFunction.forward(
                data.pos_all,
                data.node_size_x_clamped_all,
                data.node_size_y_clamped_all,
                data.offset_x_all,
                data.offset_y_all,
                data.ratio_all,
                data.bin_center_x,
                data.bin_center_y,
                data.initial_density_map,
                meta["target_density"],
                meta["xl"],
                meta["yl"],
                meta["xh"],
                meta["yh"],
                meta["bin_size_x"],
                meta["bin_size_y"],
                meta["num_movable"],
                meta["num_filler"],
                0,                          # padding = 0
                data.padding_mask,
                meta["num_bins_x"],
                meta["num_bins_y"],
                data.num_movable_impacted_bins_x,
                data.num_movable_impacted_bins_y,
                data.num_filler_impacted_bins_x,
                data.num_filler_impacted_bins_y,
                False,                      # deterministic_flag
                data.sorted_node_map,
            )
            dt_ms = (time.perf_counter() - t0) * 1e3
            if i >= args.warmup:
                cpu_cpp_times_ms.append(dt_ms)
                cpu_cpp_out = out

        cpu_cpp_ref = cpu_cpp_out.float().cpu()
        cpp_stats   = _stats(cpu_cpp_times_ms)
        print(f"  {_fmt_stats(cpp_stats)}")
        print()

    # ------------------------------------------------------------------ #
    # 2. CPU PyTorch reference (same triangle formula as C++)             #
    # ------------------------------------------------------------------ #
    print("[2/4] CPU PyTorch reference (exact triangle_density_function formula) ...")
    pt_times_ms: list = []
    pt_out: Optional[torch.Tensor] = None
    for i in range(args.warmup + args.iters):
        t0  = time.perf_counter()
        out = _cpu_pytorch_reference(data, meta)
        dt_ms = (time.perf_counter() - t0) * 1e3
        if i >= args.warmup:
            pt_times_ms.append(dt_ms)
            pt_out = out

    pt_stats = _stats(pt_times_ms)
    pt_acc   = _accuracy(pt_out.cpu(), cpu_cpp_ref)
    print(f"  {_fmt_stats(pt_stats)}")
    print(f"  vs C++: {_fmt_acc(pt_acc)}")
    print()

    # ------------------------------------------------------------------ #
    # 3. TTNN original (all ops on-device)                                #
    # ------------------------------------------------------------------ #
    tt_orig_out: Optional[torch.Tensor] = None
    tt_orig_stats: Optional[Dict] = None

    if not args.skip_ttnn_orig and tt_mod.HAS_TTNN:
        import ttnn
        print("[3/4] TTNN original (overlap + matmul on-device) ...")
        tt_orig_times_ms: list = []
        device = ttnn.open_device(device_id=0, l1_small_size=8192)
        try:
            for i in range(args.warmup + args.iters):
                t0 = time.perf_counter()
                out, _, _ = tt_mod.density_map_scatter_ttnn(
                    pos_x=data.pos_x_mf,
                    pos_y=data.pos_y_mf,
                    offset_x=data.offset_x_mf,
                    offset_y=data.offset_y_mf,
                    node_size_x_clamped=data.node_size_x_clamped_mf,
                    node_size_y_clamped=data.node_size_y_clamped_mf,
                    ratio=data.ratio_mf,
                    xl=meta["xl"],
                    yl=meta["yl"],
                    bin_size_x=meta["bin_size_x"],
                    bin_size_y=meta["bin_size_y"],
                    num_bins_x=meta["num_bins_x"],
                    num_bins_y=meta["num_bins_y"],
                    initial_density_map=data.initial_density_map,
                    chunk_size=args.chunk_size,
                    device=device,
                    return_timings=False,
                )
                ttnn.synchronize_device(device)
                dt_ms = (time.perf_counter() - t0) * 1e3
                if i >= args.warmup:
                    tt_orig_times_ms.append(dt_ms)
                    tt_orig_out = out
        finally:
            ttnn.close_device(device)

        tt_orig_stats = _stats(tt_orig_times_ms)
        tt_orig_acc   = _accuracy(tt_orig_out.cpu(), cpu_cpp_ref)
        print(f"  {_fmt_stats(tt_orig_stats)}")
        print(f"  vs C++: {_fmt_acc(tt_orig_acc)}")
    elif not tt_mod.HAS_TTNN:
        print("[3/4] TTNN original — skipped (ttnn not available)")
    else:
        print("[3/4] TTNN original — skipped (--skip-ttnn-orig)")
    print()

    # ------------------------------------------------------------------ #
    # 4. TTNN accurate (CPU overlap + TT matmul)                         #
    # ------------------------------------------------------------------ #
    tt_acc_out: Optional[torch.Tensor] = None
    tt_acc_stats: Optional[Dict] = None
    tt_acc_timings: Optional[Dict] = None
    tt_sadd_out: Optional[torch.Tensor] = None
    tt_sadd_stats: Optional[Dict] = None
    tt_v6_out: Optional[torch.Tensor] = None
    tt_v6_stats: Optional[Dict] = None

    if not args.skip_ttnn_accur and tt_mod.HAS_TTNN:
        import ttnn
        print("[4/4] TTNN accurate (CPU overlap + TT matmul) ...")
        tt_acc_times_ms: list = []
        device = ttnn.open_device(device_id=0, l1_small_size=8192)
        try:
            for i in range(args.warmup + args.iters):
                t0 = time.perf_counter()
                out, _, tms = tt_mod.density_map_scatter_ttnn_accurate(
                    pos_x=data.pos_x_mf,
                    pos_y=data.pos_y_mf,
                    offset_x=data.offset_x_mf,
                    offset_y=data.offset_y_mf,
                    node_size_x_clamped=data.node_size_x_clamped_mf,
                    node_size_y_clamped=data.node_size_y_clamped_mf,
                    ratio=data.ratio_mf,
                    xl=meta["xl"],
                    yl=meta["yl"],
                    bin_size_x=meta["bin_size_x"],
                    bin_size_y=meta["bin_size_y"],
                    num_bins_x=meta["num_bins_x"],
                    num_bins_y=meta["num_bins_y"],
                    initial_density_map=data.initial_density_map,
                    chunk_size=args.chunk_size,
                    device=device,
                    return_timings=True,
                )
                ttnn.synchronize_device(device)
                dt_ms = (time.perf_counter() - t0) * 1e3
                if i >= args.warmup:
                    tt_acc_times_ms.append(dt_ms)
                    tt_acc_out     = out
                    tt_acc_timings = tms
        finally:
            ttnn.close_device(device)

        tt_acc_stats = _stats(tt_acc_times_ms)
        tt_acc_acc   = _accuracy(tt_acc_out.cpu(), cpu_cpp_ref)
        print(f"  {_fmt_stats(tt_acc_stats)}")
        print(f"  vs C++: {_fmt_acc(tt_acc_acc)}")
        if tt_acc_timings:
            cpu_ov_ms  = tt_acc_timings.get("cpu_overlap_s", 0) * 1e3
            upload_ms  = tt_acc_timings.get("upload_s", 0) * 1e3
            tt_comp_ms = tt_acc_timings.get("tt_compute_s", 0) * 1e3
            print(f"  Phase breakdown (last iter): cpu_overlap={cpu_ov_ms:.1f} ms  "
                  f"upload={upload_ms:.1f} ms  tt_compute={tt_comp_ms:.1f} ms")
    elif not tt_mod.HAS_TTNN:
        print("[4/4] TTNN accurate — skipped (ttnn not available)")
    else:
        print("[4/4] TTNN accurate — skipped (--skip-ttnn-accur)")
    print()

    # ------------------------------------------------------------------ #
    # 5. TTNN scatter_add (native scatter path)                           #
    # ------------------------------------------------------------------ #
    if not args.skip_ttnn_scatter_add and tt_mod.HAS_TTNN:
        import ttnn
        print("[5/5] TTNN scatter_add (native scatter accumulation) ...")
        tt_sadd_times_ms: list = []
        num_bins = meta["num_bins_x"] * meta["num_bins_y"]

        t0_pre = time.perf_counter()
        index_cpu, src_cpu = _precompute_scatter_add_inputs(data, meta)
        precompute_ms = (time.perf_counter() - t0_pre) * 1e3
        print(f"  CPU precompute (index/src): {precompute_ms:.1f} ms, contributions={index_cpu.numel():,}")

        device = ttnn.open_device(device_id=0, l1_small_size=8192)
        try:
            try:
                for i in range(args.warmup + args.iters):
                    t0 = time.perf_counter()
                    out_vec = _ttnn_scatter_add_once(ttnn, device, index_cpu, src_cpu, num_bins)
                    dt_ms = (time.perf_counter() - t0) * 1e3
                    if i >= args.warmup:
                        tt_sadd_times_ms.append(dt_ms)
                        tt_sadd_out = out_vec.view(meta["num_bins_x"], meta["num_bins_y"])
            except Exception as e:
                print(f"  TTNN scatter_add unsupported/failed on this shape: {e}")
                tt_sadd_out = None
        finally:
            ttnn.close_device(device)

        if tt_sadd_out is not None and tt_sadd_times_ms:
            tt_sadd_stats = _stats(tt_sadd_times_ms)
            tt_sadd_acc = _accuracy(tt_sadd_out.cpu(), cpu_cpp_ref)
            print(f"  {_fmt_stats(tt_sadd_stats)}")
            print(f"  vs C++: {_fmt_acc(tt_sadd_acc)}")
            print("  Note: runtime excludes index/src precompute; see line above for that cost.")
    elif not tt_mod.HAS_TTNN:
        print("[5/5] TTNN scatter_add — skipped (ttnn not available)")
    else:
        print("[5/5] TTNN scatter_add — skipped (--skip-ttnn-scatter-add)")
    print()

    # ------------------------------------------------------------------ #
    # 6. TT custom v6 kernel                                               #
    # ------------------------------------------------------------------ #
    if args.run_v6_kernel and tt_mod.HAS_TTNN:
        import ttnn
        v6 = _load_tt_v6_helpers()
        if v6 is None:
            print("[6/6] TT custom v6 kernel — skipped (helper module missing)")
        else:
            print("[6/6] TT custom v6 kernel (fixed-point + wide pages) ...")
            tt_v6_times_ms: list = []
            device = ttnn.open_device(device_id=0)
            # Build [N,8] cell tensor expected by run_tt_kernel_v6.
            cell_data = torch.stack(
                [
                    data.pos_x_mf.float(),
                    data.pos_y_mf.float(),
                    data.node_size_x_clamped_mf.float(),
                    data.node_size_y_clamped_mf.float(),
                    data.offset_x_mf.float(),
                    data.offset_y_mf.float(),
                    data.ratio_mf.float(),
                    torch.zeros_like(data.ratio_mf.float()),
                ],
                dim=1,
            ).contiguous()
            try:
                for i in range(args.warmup + args.iters):
                    t0 = time.perf_counter()
                    out_tt, _ = v6.run_tt_kernel_v6(
                        device=device,
                        cell_data_torch=cell_data,
                        num_bins_x=meta["num_bins_x"],
                        num_bins_y=meta["num_bins_y"],
                        xl=meta["xl"],
                        yl=meta["yl"],
                        bsx=meta["bin_size_x"],
                        bsy=meta["bin_size_y"],
                        inv_bsx=1.0 / meta["bin_size_x"],
                        inv_bsy=1.0 / meta["bin_size_y"],
                        n_cores=v6.N_CORES_V3,
                        cells_per_page=v6.V5_CELLS_PER_PAGE,
                        scale=v6.V6_SCALE,
                    )
                    ttnn.synchronize_device(device)
                    dt_ms = (time.perf_counter() - t0) * 1e3
                    if i >= args.warmup:
                        tt_v6_times_ms.append(dt_ms)
                        tt_v6_out = v6.reduce_v3_output(
                            out_tt, v6.N_CORES_V3, meta["num_bins_x"], meta["num_bins_y"]
                        )
            except Exception as e:
                print(f"  TT v6 kernel failed/unsupported on this shape: {e}")
                tt_v6_out = None
            finally:
                ttnn.close_device(device)

            if tt_v6_out is not None and tt_v6_times_ms:
                tt_v6_stats = _stats(tt_v6_times_ms)
                tt_v6_acc = _accuracy(tt_v6_out.cpu(), cpu_cpp_ref)
                print(f"  {_fmt_stats(tt_v6_stats)}")
                print(f"  vs C++: {_fmt_acc(tt_v6_acc)}")
    elif args.run_v6_kernel and (not tt_mod.HAS_TTNN):
        print("[6/6] TT custom v6 kernel — skipped (ttnn not available)")
    print()

    # ------------------------------------------------------------------ #
    # Summary table                                                        #
    # ------------------------------------------------------------------ #
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    # Use CPU C++ as speedup baseline when available, else CPU PyTorch
    ref_ms = cpp_stats["mean"] if cpp_stats is not None else pt_stats["mean"]
    rows: list = []
    if cpp_stats is not None:
        rows.append(("CPU C++ (reference)", cpp_stats, None, 1.0))
    rows.append(("CPU PyTorch", pt_stats, pt_acc, ref_ms / max(pt_stats["mean"], 1e-9)))
    if tt_orig_stats is not None and tt_orig_out is not None:
        rows.append(("TTNN original", tt_orig_stats, _accuracy(tt_orig_out.cpu(), cpu_cpp_ref),
                     ref_ms / max(tt_orig_stats["mean"], 1e-9)))
    if tt_acc_stats is not None and tt_acc_out is not None:
        rows.append(("TTNN accurate", tt_acc_stats, _accuracy(tt_acc_out.cpu(), cpu_cpp_ref),
                     ref_ms / max(tt_acc_stats["mean"], 1e-9)))
    if tt_sadd_stats is not None and tt_sadd_out is not None:
        rows.append(("TTNN scatter_add", tt_sadd_stats, _accuracy(tt_sadd_out.cpu(), cpu_cpp_ref),
                     ref_ms / max(tt_sadd_stats["mean"], 1e-9)))
    if tt_v6_stats is not None and tt_v6_out is not None:
        rows.append(("TT custom v6", tt_v6_stats, _accuracy(tt_v6_out.cpu(), cpu_cpp_ref),
                     ref_ms / max(tt_v6_stats["mean"], 1e-9)))

    col_w = 22
    print(f"{'Path':<{col_w}} {'Mean(ms)':>9}  {'rel_l2':>10}  {'Speedup':>8}")
    print("-" * 60)
    for name, st, ac, sp in rows:
        rl2_str = f"{ac['rel_l2']:.4e}" if ac else "— (reference)"
        print(f"{name:<{col_w}} {st['mean']:9.3f}  {rl2_str:>14}  {sp:7.3f}x")
    print()

    if tt_orig_stats is not None and tt_acc_stats is not None \
            and tt_orig_out is not None and tt_acc_out is not None:
        orig_acc = _accuracy(tt_orig_out.cpu(), cpu_cpp_ref)
        new_acc  = _accuracy(tt_acc_out.cpu(),  cpu_cpp_ref)
        ratio = orig_acc["rel_l2"] / max(new_acc["rel_l2"], 1e-12)
        print(f"Accuracy improvement (TTNN accurate vs TTNN original): "
              f"rel_l2 reduced by {ratio:.2f}x")


if __name__ == "__main__":
    main()
