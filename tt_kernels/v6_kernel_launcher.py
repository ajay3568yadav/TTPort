#!/usr/bin/env python3
"""
Minimal TT custom kernel helpers for density_scatter_v6.
"""

from __future__ import annotations

import os
import pathlib
import time
from typing import Any, Dict, Tuple

import torch
import ttnn

N_CORES_V3 = 56
V5_CELLS_PER_PAGE = 64
V6_SCALE = 1024
KERNEL_FILE_PATH_V6 = "density_scatter_v6.cpp"
KERNEL_FILE_PATH_V6_ABS = str((pathlib.Path(__file__).resolve().parent / KERNEL_FILE_PATH_V6).resolve())


def _ensure_tt_metal_home() -> None:
    root = pathlib.Path(__file__).resolve().parent
    k = root / KERNEL_FILE_PATH_V6
    if k.is_file():
        os.environ["TT_METAL_HOME"] = str(root)


_ensure_tt_metal_home()


def _u32_args(args_list):
    return [int(x) & 0xFFFFFFFF for x in args_list]


def _build_kernel_runtime_args(grid_cols, grid_rows, fill_fn):
    try:
        from ttnn._ttnn.program_descriptor import VectorUInt32

        ordered = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                ordered.append(
                    (ttnn.CoreCoord(col, row), VectorUInt32(_u32_args(fill_fn(col, row))))
                )
        return ordered
    except Exception:
        return [[_u32_args(fill_fn(col, row)) for row in range(grid_rows)] for col in range(grid_cols)]


def run_tt_kernel_v6(device, cell_data_torch, num_bins_x, num_bins_y,
                     xl, yl, bsx, bsy, inv_bsx, inv_bsy,
                     n_cores=N_CORES_V3, cells_per_page=V5_CELLS_PER_PAGE,
                     scale=V6_SCALE):
    import math

    n_cells = len(cell_data_torch)
    cell_fp = cell_data_torch.float().clone()

    nx_f = cell_data_torch[:, 0] + cell_data_torch[:, 4]
    ny_f = cell_data_torch[:, 1] + cell_data_torch[:, 5]

    def to_fp_bits(t):
        return (t * scale).round().to(torch.int32).view(torch.float32)

    cell_fp[:, 0] = to_fp_bits(nx_f)
    cell_fp[:, 1] = to_fp_bits(ny_f)
    cell_fp[:, 2] = to_fp_bits(cell_data_torch[:, 2])
    cell_fp[:, 3] = to_fp_bits(cell_data_torch[:, 3])
    cell_fp[:, 4] = torch.zeros(n_cells)
    cell_fp[:, 5] = torch.zeros(n_cells)
    # Keep ratio in real units.
    # Kernel reconstructs nx/ny/szx/szy from fixed-point, then computes overlap
    # in real coordinates; dividing ratio by scale^2 here would undercount area.
    cell_fp[:, 6] = cell_data_torch[:, 6].float()

    t_upload_start = time.perf_counter()
    MAX_GRID_X, MAX_GRID_Y = 8, 7
    n_cores = min(n_cores, MAX_GRID_X * MAX_GRID_Y)
    grid_cols = min(n_cores, MAX_GRID_X)
    grid_rows = min(math.ceil(n_cores / grid_cols), MAX_GRID_Y)
    actual_cores = grid_cols * grid_rows

    chunk = actual_cores * cells_per_page
    n_padded = math.ceil(n_cells / chunk) * chunk
    if n_padded > n_cells:
        pad = torch.zeros(n_padded - n_cells, 8, dtype=torch.float32)
        cell_wide = torch.cat([cell_fp, pad], dim=0)
    else:
        cell_wide = cell_fp
    n_pages = n_padded // cells_per_page
    cell_wide = cell_wide.view(n_pages, cells_per_page * 8).contiguous()

    cell_tt = ttnn.from_torch(cell_wide, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_tt = ttnn.allocate_tensor_on_device(
        ttnn.Shape([actual_cores * num_bins_x, num_bins_y]), ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(device)
    t_upload_ms = (time.perf_counter() - t_upload_start) * 1e3

    cell_addr = cell_tt.buffer_address()
    out_addr = out_tt.buffer_address()
    cell_ct_args = ttnn.TensorAccessorArgs(cell_tt).get_compile_time_args()
    out_ct_args = ttnn.TensorAccessorArgs(out_tt).get_compile_time_args()
    compile_time_args = cell_ct_args + out_ct_args

    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_cols - 1, grid_rows - 1))])
    grid_size_bytes = num_bins_x * num_bins_y * 4
    cell_page_bytes_py = cells_per_page * 8 * 4

    grid_cb_desc = ttnn.CBDescriptor(
        total_size=grid_size_bytes, core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=grid_size_bytes)])
    cell_cb_desc = ttnn.CBDescriptor(
        total_size=cell_page_bytes_py, core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.float32, page_size=cell_page_bytes_py)])

    def to_int32_bits(v):
        return int(round(v * scale)) & 0xFFFFFFFF

    pages_per_core = n_pages // actual_cores
    common_args = [cell_addr, out_addr, n_cells, num_bins_x, num_bins_y, num_bins_y * 4,
                   to_int32_bits(xl), to_int32_bits(yl), to_int32_bits(bsx), to_int32_bits(bsy)]

    def _fill_v6(col, row):
        cid = row * grid_cols + col
        page_start = cid * pages_per_core
        page_end = page_start + pages_per_core
        out_row_offset = cid * num_bins_x
        return common_args + [page_start, page_end, out_row_offset]

    runtime_args = _build_kernel_runtime_args(grid_cols, grid_rows, _fill_v6)

    kernel_desc = ttnn.KernelDescriptor(
        kernel_source=KERNEL_FILE_PATH_V6_ABS,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=compile_time_args,
        runtime_args=runtime_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    program_desc = ttnn.ProgramDescriptor(kernels=[kernel_desc], semaphores=[], cbs=[grid_cb_desc, cell_cb_desc])

    t_compute_start = time.perf_counter()
    result = ttnn.generic_op([cell_tt, out_tt], program_desc)
    ttnn.synchronize_device(device)
    t_compute_ms = (time.perf_counter() - t_compute_start) * 1e3
    return result, {
        "upload_ms": t_upload_ms,
        "compute_ms": t_compute_ms,
        "total_ms": t_upload_ms + t_compute_ms,
        "n_cores": actual_cores,
    }


def reduce_v3_output(out_tt_tensor, n_cores, num_bins_x, num_bins_y):
    out_torch = ttnn.to_torch(out_tt_tensor).float()
    rows = int(out_torch.shape[0])
    actual_cores = rows // num_bins_x
    out_3d = out_torch.view(actual_cores, num_bins_x, num_bins_y)
    return out_3d.sum(dim=0)

