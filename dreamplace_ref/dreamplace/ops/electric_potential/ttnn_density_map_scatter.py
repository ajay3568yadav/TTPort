##
# @file   ttnn_density_map_scatter.py
# @brief  Density map scatter using TTNN matrix/vector ops (no scatter_add).
# Because ttnn.scatter_add has a row length limit of 256 elements, we use
# matmul + add instead. This path requires CHUNKING over cells so Px/Py fit
# in device memory. The custom v6 (and v6_stripe) Metal kernels do NOT need
# chunking: they stream all cells and accumulate per-core partial grids.
#
# Algorithm: Px = overlap in x, Py = overlap in y; then
#   movable_contribution = Px.T @ (Py * ratio); density_map = initial + movable.
# Real scale: process cells in chunks; each chunk uses TTNN for overlap + matmul.
#
# Overlap formula (abs+clip — mathematically equivalent, fewer ops):
#   Px[i,k] = clip(half_span_x[i] - |cell_cx[i] - bin_cx[k]|, 0, bin_size_x)
#   where half_span_x = (sx + bin_size_x)/2, cell_cx = node_x + sx/2.
#   Valid because DREAMPlace clamps sx >= sqrt(2)*bin_size > bin_size (always).
#   Uses 4 TTNN ops (subtract+abs+subtract+clip) vs 7 ops with the original
#   add/le/where/ge/where/subtract/relu formulation.
#
# CAN on TTNN: add, subtract, multiply, abs, clip, relu, transpose, matmul,
#   tilize, untilize, from_torch, to_torch, deallocate.
# CANNOT on TTNN: ttnn.scatter_add (256-element row limit); we use matmul + add.
#

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# TT tile size; tilize/matmul expect dimensions multiple of this.
TT_TILE_SIZE = 32

try:
    import ttnn
    HAS_TTNN = True
except ImportError:
    HAS_TTNN = False


def _ttnn_px_overlap(ccx_tt, bc_x_tt, hsx_tt, bin_size: float):
    """
    Compute x-overlap for all (cell, bin) pairs using the abs+clip formula:
        Px[i,k] = clip(half_span[i] - |cell_center[i] - bin_center[k]|, 0, bin_size)

    Requires 4 TTNN ops instead of 7 with the min/max/relu formulation.
    Valid only when sx >= bin_size for every cell (DREAMPlace clamps sx >= sqrt(2)*bin_size).

    Args:
        ccx_tt:  (n, 1) cell center x on device.
        bc_x_tt: (1, K) bin center x on device.
        hsx_tt:  (n, 1) half_span_x = (sx + bin_size_x)/2 on device.
        bin_size: scalar upper bound (bin_size_x or bin_size_y).
    Returns:
        px_tt: (n, K) overlap tensor (ROW_MAJOR_LAYOUT).
    """
    dist = ttnn.abs(ttnn.subtract(ccx_tt, bc_x_tt))   # (n,K): |cell_cx - bin_cx|
    pre = ttnn.subtract(hsx_tt, dist)                  # (n,K): half_span - dist
    ttnn.deallocate(dist)
    px = ttnn.clip(pre, 0.0, bin_size)                 # (n,K): clip to [0, bin_size]
    ttnn.deallocate(pre)
    return px


def density_map_scatter_ttnn(
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    offset_x: torch.Tensor,
    offset_y: torch.Tensor,
    node_size_x_clamped: torch.Tensor,
    node_size_y_clamped: torch.Tensor,
    ratio: torch.Tensor,
    xl: float,
    yl: float,
    bin_size_x: float,
    bin_size_y: float,
    num_bins_x: int,
    num_bins_y: int,
    initial_density_map: torch.Tensor,
    device_id: int = 0,
    chunk_size: int = 32768,
    tt_dtype=None,
    return_timings: bool = False,
    timing_log_path: Optional[str] = None,
    device: Optional[Any] = None,
    return_on_device: bool = False,
) -> Tuple[Any, Optional[object], Optional[Dict[str, float]]]:
    """
    Compute full density map (initial + movable scatter) using TTNN matrix ops.

    Overlap formula (abs+clip — 4 ops per axis):
        Px[i,k] = clip(half_span_x[i] - |cell_cx[i] - bin_cx[k]|, 0, bin_size_x)
    where cell_cx = node_x + sx/2, half_span_x = (sx + bin_size_x)/2.
    Accumulates density_map += Px.T @ (Py * ratio) per chunk.

    SCATTER-ADD is NOT used; accumulation is done via matmul + add.

    Args:
        pos_x, pos_y: (C,) cell positions (left, bottom).
        offset_x, offset_y: (C,) offsets for stretched model.
        node_size_x_clamped, node_size_y_clamped: (C,) clamped sizes (>= sqrt(2)*bin_size).
        ratio: (C,) area ratio per cell.
        xl, yl: canvas origin.
        bin_size_x, bin_size_y: bin dimensions.
        num_bins_x, num_bins_y: grid size (K, H).
        initial_density_map: (K, H) fixed-cell map.
        device_id: TT device ID (used only if device is None).
        chunk_size: cells per chunk (default 32768 for fewer kernel dispatches).
        tt_dtype: ttnn dtype (default float32 for correctness).
        timing_log_path: if set, write per-phase duration (ms) to this CSV for optimization analysis.
        device: if provided, use this TT device (do not open/close); enables zero-copy to DCT solver.
        return_on_device: if True, return (density_map_tt, device) without downloading; density is
            raw (unnormalized); caller can pass to TTNNFieldSolver.solve_from_device(., bin_area).

    Returns:
        density_map: (num_bins_x, num_bins_y) — torch on CPU if not return_on_device, else ttnn tensor on device.
        device: TT device (caller must close if this function opened it).
        timings: If return_timings=True, dict with "upload_s", "compute_s" (download excluded).
    """
    if not HAS_TTNN:
        logger.warning("ttnn not available; cannot run TTNN density scatter")
        return initial_density_map.clone(), None, None

    if tt_dtype is None:
        # float32 is required: layout coordinates for real designs span ~10k units,
        # where bfloat16 ULP (~64) is larger than a single bin width (~20), causing
        # catastrophically wrong density maps. bfloat16 is only safe if coordinates
        # are pre-normalized to a small range (e.g. [0, 1]).
        tt_dtype = ttnn.float32

    C = pos_x.shape[0]
    K, H = num_bins_x, num_bins_y

    # --- Precompute per-cell arrays once (avoids repeated work inside the loop) ---
    # Using abs+clip formula: Px[i,k] = clip(half_span_x[i]-|cell_cx[i]-bin_cx[k]|, 0, bsz_x)
    node_x_all = (pos_x + offset_x).float()                       # (C,)
    node_y_all = (pos_y + offset_y).float()                       # (C,)
    sx_all = node_size_x_clamped.float()                           # (C,)
    sy_all = node_size_y_clamped.float()                           # (C,)
    cell_cx_all = node_x_all + 0.5 * sx_all                       # (C,) cell center x
    cell_cy_all = node_y_all + 0.5 * sy_all                       # (C,) cell center y
    half_span_x_all = 0.5 * (sx_all + bin_size_x)                 # (C,) half total span x
    half_span_y_all = 0.5 * (sy_all + bin_size_y)                 # (C,) half total span y
    ratio_all = ratio.float()                                      # (C,)

    # --- Bin center vectors (1, K) and (1, H) — uploaded once, reused every chunk ---
    bin_center_x = torch.tensor(
        xl + (np.arange(K, dtype=np.float32) + 0.5) * bin_size_x,
        dtype=torch.float32, device=pos_x.device,
    ).reshape(1, K)
    bin_center_y = torch.tensor(
        yl + (np.arange(H, dtype=np.float32) + 0.5) * bin_size_y,
        dtype=torch.float32, device=pos_x.device,
    ).reshape(1, H)

    owned_device = device is None
    if device is None:
        device = ttnn.open_device(device_id=device_id, l1_small_size=8192)

    timings: Dict[str, float] = {"upload_s": 0.0, "compute_s": 0.0}

    # Per-phase timing log for optimization (CSV: phase,duration_ms,chunk_index)
    timing_entries: List[Tuple[str, float, Optional[int]]] = []

    def _end_phase(phase_name: str, t0: float, chunk_ix: Optional[int] = None) -> None:
        if timing_log_path is None:
            return
        ttnn.synchronize_device(device)
        ms = (time.perf_counter() - t0) * 1000.0
        timing_entries.append((phase_name, ms, chunk_ix))

    def to_tt(t: torch.Tensor, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(
            t.detach().float().contiguous(),
            dtype=tt_dtype, layout=layout, device=device,
        )

    def to_tt_timed(t: torch.Tensor, layout=ttnn.ROW_MAJOR_LAYOUT):
        t0 = time.perf_counter()
        out = to_tt(t, layout=layout)
        if return_timings:
            timings["upload_s"] += time.perf_counter() - t0
        return out

    _upload = to_tt_timed if return_timings else to_tt

    # Upload bin centers (1, K) and (1, H) - reused every chunk
    if timing_log_path:
        t0_bin = time.perf_counter()
    bc_x_tt = _upload(bin_center_x)
    bc_y_tt = _upload(bin_center_y)
    if timing_log_path:
        _end_phase("upload_bin_centers", t0_bin)

    # Accumulator: start from initial_density_map (K, H). Use 2D (K, H).
    # matmul and add require TILE layout; keep acc tilized for the whole loop.
    if timing_log_path:
        t0_init = time.perf_counter()
    init_cpu = initial_density_map.float().contiguous()
    if init_cpu.dim() != 2:
        init_cpu = init_cpu.view(K, H)
    acc_rm = _upload(init_cpu)
    acc = ttnn.tilize(acc_rm)
    ttnn.deallocate(acc_rm)
    if timing_log_path:
        _end_phase("upload_init_tilize", t0_init)

    if return_timings:
        t_compute_start = time.perf_counter()

    num_chunks = (C + chunk_size - 1) // chunk_size
    for chunk_ix, start in enumerate(range(0, C, chunk_size)):
        end = min(start + chunk_size, C)
        n = end - start
        # Pad n to multiple of TT_TILE_SIZE so tilize/matmul succeed (last chunk may have n%32 != 0).
        n_pad = ((n + TT_TILE_SIZE - 1) // TT_TILE_SIZE) * TT_TILE_SIZE

        zeros_col = torch.zeros(n_pad - n, 1, dtype=torch.float32, device=pos_x.device)

        def _slice_col(arr1d: torch.Tensor) -> torch.Tensor:
            sliced = arr1d[start:end].reshape(n, 1).float()
            if n_pad != n:
                return torch.cat([sliced, zeros_col], dim=0)
            return sliced

        ccx = _slice_col(cell_cx_all)       # (n_pad, 1) cell center x
        ccy = _slice_col(cell_cy_all)       # (n_pad, 1) cell center y
        hsx = _slice_col(half_span_x_all)   # (n_pad, 1) half span x
        hsy = _slice_col(half_span_y_all)   # (n_pad, 1) half span y
        r   = _slice_col(ratio_all)         # (n_pad, 1) area ratio
        n = n_pad

        # Upload per-chunk cell vectors (n, 1)
        if timing_log_path:
            t0_up = time.perf_counter()
        ccx_tt = _upload(ccx)
        ccy_tt = _upload(ccy)
        hsx_tt = _upload(hsx)
        hsy_tt = _upload(hsy)
        r_tt   = _upload(r)
        if timing_log_path:
            _end_phase("chunk_upload", t0_up, chunk_ix)

        # Px (n, K) via abs+clip: clip(half_span_x - |cell_cx - bin_cx|, 0, bin_size_x)
        # 4 TTNN ops: subtract (broadcast n×K), abs, subtract, clip
        if timing_log_path:
            t0_px = time.perf_counter()
        px_tt = _ttnn_px_overlap(ccx_tt, bc_x_tt, hsx_tt, bin_size_x)
        ttnn.deallocate(ccx_tt)
        ttnn.deallocate(hsx_tt)
        if timing_log_path:
            _end_phase("chunk_px_compute", t0_px, chunk_ix)

        # Py (n, H) via abs+clip: same formula along y axis
        if timing_log_path:
            t0_py = time.perf_counter()
        py_tt = _ttnn_px_overlap(ccy_tt, bc_y_tt, hsy_tt, bin_size_y)
        ttnn.deallocate(ccy_tt)
        ttnn.deallocate(hsy_tt)
        if timing_log_path:
            _end_phase("chunk_py_compute", t0_py, chunk_ix)

        # py_ratio = Py * ratio (n, H) * (n, 1) -> (n, H)
        if timing_log_path:
            t0_ratio = time.perf_counter()
        py_ratio_tt = ttnn.multiply(py_tt, r_tt)
        ttnn.deallocate(py_tt)
        ttnn.deallocate(r_tt)
        if timing_log_path:
            _end_phase("chunk_py_ratio", t0_ratio, chunk_ix)

        # movable_contribution = Px.T @ (Py * ratio)  -> (K, n) @ (n, H) = (K, H)
        # matmul requires TILE layout; tilize both operands.
        if timing_log_path:
            t0_trans = time.perf_counter()
        px_T = ttnn.transpose(px_tt, 0, 1)  # (K, n)
        if timing_log_path:
            _end_phase("chunk_transpose", t0_trans, chunk_ix)

        if timing_log_path:
            t0_til = time.perf_counter()
        px_T_tiled = ttnn.tilize(px_T)
        py_ratio_tiled = ttnn.tilize(py_ratio_tt)
        if timing_log_path:
            _end_phase("chunk_tilize", t0_til, chunk_ix)

        if timing_log_path:
            t0_mat = time.perf_counter()
        contrib_tt = ttnn.matmul(px_T_tiled, py_ratio_tiled)  # (K, n) @ (n, H) = (K, H)
        if timing_log_path:
            _end_phase("chunk_matmul", t0_mat, chunk_ix)

        if timing_log_path:
            t0_add = time.perf_counter()
        # Accumulate: acc += contrib_tt (TTNN add; scatter_add NOT used - see comment below)
        new_acc = ttnn.add(acc, contrib_tt)
        ttnn.deallocate(acc)
        acc = new_acc
        if timing_log_path:
            _end_phase("chunk_add", t0_add, chunk_ix)

        if timing_log_path:
            t0_dealloc = time.perf_counter()
        # Free chunk tensors to save device memory
        ttnn.deallocate(px_tt)
        ttnn.deallocate(py_ratio_tt)
        ttnn.deallocate(px_T)
        ttnn.deallocate(px_T_tiled)
        ttnn.deallocate(py_ratio_tiled)
        ttnn.deallocate(contrib_tt)
        if timing_log_path:
            _end_phase("chunk_deallocate", t0_dealloc, chunk_ix)

    # -------------------------------------------------------------------------
    # SCATTER-ADD: CANNOT use ttnn.scatter_add here.
    #   - ttnn.scatter_add has a hardware limit: scatter row length <= 256 elements.
    #   - Density map scatter: each "row" of sources can be long (many cells per bin),
    #     and we need to add into a (K, H) grid from C cells; the index pattern
    #     does not fit the 256 limit for real-scale C.
    #   - We therefore use the mathematically equivalent matmul accumulation:
    #     density_map += Px.T @ (Py * ratio) per chunk, which uses only ttnn.matmul
    #     and ttnn.add.
    # -------------------------------------------------------------------------

    if return_timings:
        timings["compute_s"] = time.perf_counter() - t_compute_start

    ttnn.synchronize_device(device)

    # Cleanup bin center tensors (no longer needed after loop)
    ttnn.deallocate(bc_x_tt)
    ttnn.deallocate(bc_y_tt)

    if return_on_device:
        # Return density on device (TILE layout, unnormalized) for direct feed to DCT solver.
        if timing_log_path and timing_entries:
            with open(timing_log_path, "w") as log_file:
                log_file.write("phase,duration_ms,chunk_index\n")
                for phase_name, ms, cix in timing_entries:
                    log_file.write(f"{phase_name},{ms:.3f},{cix if cix is not None else ''}\n")
                by_phase: Dict[str, float] = defaultdict(float)
                for phase_name, ms, _ in timing_entries:
                    by_phase[phase_name] += ms
                total_ms = sum(by_phase.values())
                log_file.write("\n# Summary (total ms by phase)\n")
                log_file.write("phase,total_ms,pct\n")
                for phase_name in sorted(by_phase.keys()):
                    t = by_phase[phase_name]
                    pct = 100.0 * t / total_ms if total_ms > 0 else 0.0
                    log_file.write(f"{phase_name},{t:.3f},{pct:.1f}\n")
                log_file.write(f"TOTAL,{total_ms:.3f},100.0\n")
            logger.info("TTNN density scatter timing log written to %s", timing_log_path)
        if return_timings:
            return acc, device, timings
        return acc, device, None

    # acc is in TILE layout; untilize before host copy
    if timing_log_path:
        t0_until = time.perf_counter()
    acc_rm = ttnn.untilize(acc)
    ttnn.deallocate(acc)
    if timing_log_path:
        _end_phase("untilize_acc", t0_until)
    if timing_log_path:
        t0_dl = time.perf_counter()
    density_map = ttnn.to_torch(acc_rm)
    ttnn.deallocate(acc_rm)
    if timing_log_path:
        _end_phase("to_torch", t0_dl)
    if density_map.dim() == 3:
        density_map = density_map.squeeze(0)
    density_map = density_map[:num_bins_x, :num_bins_y].float().contiguous()

    if timing_log_path and timing_entries:
        with open(timing_log_path, "w") as log_file:
            log_file.write("phase,duration_ms,chunk_index\n")
            for phase_name, ms, cix in timing_entries:
                log_file.write(f"{phase_name},{ms:.3f},{cix if cix is not None else ''}\n")
            by_phase: Dict[str, float] = defaultdict(float)
            for phase_name, ms, _ in timing_entries:
                by_phase[phase_name] += ms
            total_ms = sum(by_phase.values())
            log_file.write("\n# Summary (total ms by phase; use for optimization focus)\n")
            log_file.write("phase,total_ms,pct\n")
            for phase_name in sorted(by_phase.keys()):
                t = by_phase[phase_name]
                pct = 100.0 * t / total_ms if total_ms > 0 else 0.0
                log_file.write(f"{phase_name},{t:.3f},{pct:.1f}\n")
            log_file.write(f"TOTAL,{total_ms:.3f},100.0\n")
        logger.info("TTNN density scatter timing log written to %s (summary: %s)", timing_log_path, dict(by_phase))

    if return_timings:
        return density_map, device, timings
    return density_map, device, None


def density_map_scatter_ttnn_accurate(
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    offset_x: torch.Tensor,
    offset_y: torch.Tensor,
    node_size_x_clamped: torch.Tensor,
    node_size_y_clamped: torch.Tensor,
    ratio: torch.Tensor,
    xl: float,
    yl: float,
    bin_size_x: float,
    bin_size_y: float,
    num_bins_x: int,
    num_bins_y: int,
    initial_density_map: torch.Tensor,
    device_id: int = 0,
    chunk_size: int = 32768,
    tt_dtype=None,
    return_timings: bool = False,
    device: Optional[Any] = None,
) -> Tuple[torch.Tensor, Optional[object], Optional[Dict[str, float]]]:
    """
    Accurate density-map scatter: CPU overlap + TTNN matmul.

    Computes Px (n, K) and Py (n, H) on CPU using the exact DREAMPlace
    ``triangle_density_function`` formula from ``density_function.h``::

        Px[i, k] = max(0, min(eff_x[i] + sx[i], xl + (k+1)*bin_size_x)
                          - max(eff_x[i], xl + k*bin_size_x))

    where ``eff_x = pos_x + offset_x`` and ``sx = node_size_x_clamped``.

    The CPU overlap computation is bit-for-bit identical to the C++ kernel for
    overlapping (bin, cell) pairs. Clips negative values to 0 (the C++ kernel
    achieves the same by only iterating over bins in the cell's bounding box).

    After computing the overlaps on CPU, chunked TTNN matmuls accumulate::

        density_map += Px_T_chunk @ py_ratio_chunk   (K x H)

    This hybrid approach eliminates precision loss that arises when the
    subtract/abs/clip overlap ops run on-device in the original path.

    Args:
        pos_x, pos_y: (C,) cell left/bottom positions.
        offset_x, offset_y: (C,) offsets from stretching (≤ 0 for movable cells).
        node_size_x_clamped, node_size_y_clamped: (C,) sizes clamped to sqrt(2)*bin_size.
        ratio: (C,) original_area / stretched_area per cell.
        xl, yl: placement region origin.
        bin_size_x, bin_size_y: bin dimensions.
        num_bins_x, num_bins_y: grid shape (K, H).
        initial_density_map: (K, H) fixed-cell contribution (CPU tensor).
        device_id: TT device index (used only if ``device`` is None).
        chunk_size: cells processed per TT matmul dispatch.
        tt_dtype: TTNN dtype (default float32).
        return_timings: if True, return timing dict with
            ``cpu_overlap_s``, ``upload_s``, ``tt_compute_s``.
        device: pre-opened TT device to reuse (caller owns it); opened
            internally if None.

    Returns:
        density_map: (K, H) CPU float32 tensor.
        device: TT device handle (same as input if provided, else newly opened).
        timings: dict or None.
    """
    if not HAS_TTNN:
        logger.warning("ttnn not available; falling back to initial_density_map")
        return initial_density_map.clone(), None, None

    if tt_dtype is None:
        tt_dtype = ttnn.float32

    C = pos_x.shape[0]
    K, H = num_bins_x, num_bins_y

    # ---- CPU pre-computation: effective positions + bin edge vectors ----
    eff_x = (pos_x + offset_x).float()          # (C,) effective left edge
    eff_y = (pos_y + offset_y).float()
    sx    = node_size_x_clamped.float()          # (C,)
    sy    = node_size_y_clamped.float()
    r     = ratio.float()                        # (C,)

    # Bin left / right edges — computed as xl + k*bin_size to match C++ exactly.
    bin_left_x  = torch.tensor(
        xl + np.arange(K, dtype=np.float32) * bin_size_x, dtype=torch.float32
    )                                             # (K,)
    bin_right_x = bin_left_x + float(bin_size_x) # (K,)
    bin_left_y  = torch.tensor(
        yl + np.arange(H, dtype=np.float32) * bin_size_y, dtype=torch.float32
    )                                             # (H,)
    bin_right_y = bin_left_y + float(bin_size_y) # (H,)

    owned_device = device is None
    if device is None:
        device = ttnn.open_device(device_id=device_id, l1_small_size=8192)

    timings: Dict[str, float] = {"cpu_overlap_s": 0.0, "upload_s": 0.0, "tt_compute_s": 0.0}

    def to_tt(t: torch.Tensor) -> Any:
        return ttnn.from_torch(
            t.detach().float().contiguous(),
            dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
        )

    # Upload + tilize initial density map once.
    acc_rm = to_tt(initial_density_map.float().view(K, H))
    acc    = ttnn.tilize(acc_rm)
    ttnn.deallocate(acc_rm)

    for start in range(0, C, chunk_size):
        end = min(start + chunk_size, C)
        n   = end - start
        # Pad n to multiple of TT_TILE_SIZE for tilize / matmul.
        n_pad = ((n + TT_TILE_SIZE - 1) // TT_TILE_SIZE) * TT_TILE_SIZE

        # ---- CPU: exact triangle_density_function overlap ----
        t0_cpu = time.perf_counter()

        ex_c = eff_x[start:end].view(n, 1)   # (n, 1)
        ey_c = eff_y[start:end].view(n, 1)
        sx_c = sx[start:end].view(n, 1)
        sy_c = sy[start:end].view(n, 1)
        r_c  = r[start:end].view(n, 1)

        # Px[i,k] = max(0, min(eff_x[i]+sx[i], bin_right[k]) - max(eff_x[i], bin_left[k]))
        Px = torch.clamp(
            torch.minimum(ex_c + sx_c, bin_right_x.view(1, K))
            - torch.maximum(ex_c, bin_left_x.view(1, K)),
            min=0.0,
        )  # (n, K)

        Py = torch.clamp(
            torch.minimum(ey_c + sy_c, bin_right_y.view(1, H))
            - torch.maximum(ey_c, bin_left_y.view(1, H)),
            min=0.0,
        )  # (n, H)

        py_ratio = (Py * r_c).contiguous()  # (n, H)

        # Px.T: (K, n) — transpose on CPU, then pad n → n_pad with zeros.
        px_T = Px.t().contiguous()          # (K, n)
        if n_pad != n:
            zeros_k = torch.zeros(K, n_pad - n, dtype=torch.float32)
            zeros_h = torch.zeros(n_pad - n, H, dtype=torch.float32)
            px_T     = torch.cat([px_T, zeros_k], dim=1)      # (K, n_pad)
            py_ratio = torch.cat([py_ratio, zeros_h], dim=0)  # (n_pad, H)

        if return_timings:
            timings["cpu_overlap_s"] += time.perf_counter() - t0_cpu

        # ---- Upload to TT ----
        t0_up = time.perf_counter()
        px_T_tt      = to_tt(px_T)       # (K, n_pad)
        py_ratio_tt  = to_tt(py_ratio)   # (n_pad, H)
        if return_timings:
            timings["upload_s"] += time.perf_counter() - t0_up

        # ---- TT matmul: (K, n_pad) @ (n_pad, H) = (K, H) ----
        t0_tt = time.perf_counter()
        px_T_tiled      = ttnn.tilize(px_T_tt)
        py_ratio_tiled  = ttnn.tilize(py_ratio_tt)
        ttnn.deallocate(px_T_tt)
        ttnn.deallocate(py_ratio_tt)

        contrib_tt = ttnn.matmul(px_T_tiled, py_ratio_tiled)
        ttnn.deallocate(px_T_tiled)
        ttnn.deallocate(py_ratio_tiled)

        new_acc = ttnn.add(acc, contrib_tt)
        ttnn.deallocate(acc)
        ttnn.deallocate(contrib_tt)
        acc = new_acc

        if return_timings:
            timings["tt_compute_s"] += time.perf_counter() - t0_tt

    ttnn.synchronize_device(device)

    # ---- Download result ----
    acc_rm      = ttnn.untilize(acc)
    ttnn.deallocate(acc)
    density_map = ttnn.to_torch(acc_rm).float()
    ttnn.deallocate(acc_rm)
    if density_map.dim() == 3:
        density_map = density_map.squeeze(0)
    density_map = density_map[:K, :H].contiguous()

    if return_timings:
        return density_map, device, timings
    return density_map, device, None


def density_map_scatter_ttnn_full_matrices(
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    offset_x: torch.Tensor,
    offset_y: torch.Tensor,
    node_size_x_clamped: torch.Tensor,
    node_size_y_clamped: torch.Tensor,
    ratio: torch.Tensor,
    xl: float,
    yl: float,
    bin_size_x: float,
    bin_size_y: float,
    num_bins_x: int,
    num_bins_y: int,
    initial_density_map: torch.Tensor,
    device_id: int = 0,
    tt_dtype=None,
    return_timings: bool = False,
) -> Tuple[torch.Tensor, Optional[object], Optional[Dict[str, float]]]:
    """
    Same algorithm but builds full Px (C x K) and Py (C x H) and does
    one matmul: density_map = initial + Px.T @ (Py * ratio). No chunking.

    Use only when C is small enough that Px and Py fit in device memory
    (e.g. C <= 16k, K=H=512 -> Px 8M floats ~32 MB). For real scale (C=200k+)
    use density_map_scatter_ttnn() which chunks over cells.
    Matmul inputs are tilized; result is untilized before host copy.
    """
    if not HAS_TTNN:
        return initial_density_map.clone(), None, None

    if tt_dtype is None:
        tt_dtype = ttnn.float32

    C_orig = pos_x.shape[0]
    K, H = num_bins_x, num_bins_y
    # Pad C to multiple of TT_TILE_SIZE for tilize/matmul
    C = ((C_orig + TT_TILE_SIZE - 1) // TT_TILE_SIZE) * TT_TILE_SIZE
    if C != C_orig:
        pad = C - C_orig
        pos_x = torch.cat([pos_x, torch.zeros(pad, device=pos_x.device, dtype=pos_x.dtype)])
        pos_y = torch.cat([pos_y, torch.zeros(pad, device=pos_y.device, dtype=pos_y.dtype)])
        offset_x = torch.cat([offset_x, torch.zeros(pad, device=offset_x.device, dtype=offset_x.dtype)])
        offset_y = torch.cat([offset_y, torch.zeros(pad, device=offset_y.device, dtype=offset_y.dtype)])
        node_size_x_clamped = torch.cat([node_size_x_clamped, torch.zeros(pad, device=pos_x.device, dtype=torch.float32)])
        node_size_y_clamped = torch.cat([node_size_y_clamped, torch.zeros(pad, device=pos_x.device, dtype=torch.float32)])
        ratio = torch.cat([ratio, torch.zeros(pad, device=ratio.device, dtype=ratio.dtype)])

    timings: Dict[str, float] = {"upload_s": 0.0, "compute_s": 0.0}
    t0_upload = time.perf_counter() if return_timings else None

    # Build Px, Py on CPU
    node_x = (pos_x + offset_x).reshape(C, 1).float()
    node_y = (pos_y + offset_y).reshape(C, 1).float()
    sx = node_size_x_clamped.reshape(C, 1).float()
    sy = node_size_y_clamped.reshape(C, 1).float()

    bin_left_x = torch.tensor(
        xl + np.arange(K, dtype=np.float32) * bin_size_x,
        dtype=torch.float32, device=pos_x.device,
    ).reshape(1, K)
    bin_right_x = torch.tensor(
        xl + (np.arange(K, dtype=np.float32) + 1) * bin_size_x,
        dtype=torch.float32, device=pos_x.device,
    ).reshape(1, K)
    bin_left_y = torch.tensor(
        yl + np.arange(H, dtype=np.float32) * bin_size_y,
        dtype=torch.float32, device=pos_x.device,
    ).reshape(1, H)
    bin_right_y = torch.tensor(
        yl + (np.arange(H, dtype=np.float32) + 1) * bin_size_y,
        dtype=torch.float32, device=pos_x.device,
    ).reshape(1, H)

    px_cpu = torch.relu(
        torch.minimum(node_x + sx, bin_right_x) - torch.maximum(node_x, bin_left_x)
    )
    py_cpu = torch.relu(
        torch.minimum(node_y + sy, bin_right_y) - torch.maximum(node_y, bin_left_y)
    )
    py_ratio_cpu = (py_cpu * ratio.reshape(C, 1)).float().contiguous()

    device = ttnn.open_device(device_id=device_id, l1_small_size=8192)

    def to_tt(t):
        return ttnn.from_torch(
            t.detach().float().contiguous(),
            dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
        )

    # Upload full Px (C, K), Py*ratio (C, H), initial (K, H). WARNING: large C may OOM.
    init_cpu = initial_density_map.float().contiguous().view(K, H)
    px_tt = to_tt(px_cpu)
    py_ratio_tt = to_tt(py_ratio_cpu)
    init_rm = to_tt(init_cpu)
    init_tt = ttnn.tilize(init_rm)
    ttnn.deallocate(init_rm)

    if return_timings:
        timings["upload_s"] = time.perf_counter() - t0_upload
        t_compute_start = time.perf_counter()

    # density_map = initial + Px.T @ (Py * ratio); matmul requires TILE layout
    px_T_tt = ttnn.transpose(px_tt, 0, 1)  # (K, C)
    px_T_tiled = ttnn.tilize(px_T_tt)
    py_ratio_tiled = ttnn.tilize(py_ratio_tt)
    ttnn.deallocate(px_tt)
    ttnn.deallocate(py_ratio_tt)
    ttnn.deallocate(px_T_tt)

    contrib_tt = ttnn.matmul(px_T_tiled, py_ratio_tiled)  # (K, C) @ (C, H) = (K, H)
    ttnn.deallocate(px_T_tiled)
    ttnn.deallocate(py_ratio_tiled)

    result_tt = ttnn.add(init_tt, contrib_tt)
    ttnn.deallocate(init_tt)
    ttnn.deallocate(contrib_tt)

    if return_timings:
        timings["compute_s"] = time.perf_counter() - t_compute_start

    ttnn.synchronize_device(device)
    result_rm = ttnn.untilize(result_tt)
    ttnn.deallocate(result_tt)
    density_map = ttnn.to_torch(result_rm).float()[:num_bins_x, :num_bins_y].contiguous()
    ttnn.deallocate(result_rm)

    if return_timings:
        return density_map, device, timings
    return density_map, device, None
