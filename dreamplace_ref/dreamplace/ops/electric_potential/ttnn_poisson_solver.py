##
# @file   ttnn_poisson_solver.py
# @brief  DCT-based Poisson field solver on Tenstorrent hardware via ttnn.
#
# Replaces the FFT-based DCT/IDXST/IDCT pipeline in electric_potential.py
# with dense matrix multiplications executed on Tenstorrent Tensix cores.
#
# The math is identical to the CPU path:
#   1. auv        = DCT2(density_map)               — 2 matmuls
#   2. fx_auv     = auv * wu_by_wu2_plus_wv2_half   — pointwise
#   3. fy_auv     = auv * wv_by_wu2_plus_wv2_half   — pointwise
#   4. field_x    = IDXST_IDCT(fx_auv)              — 2 matmuls
#   5. field_y    = IDCT_IDXST(fy_auv)              — 2 matmuls
# Total: 6 matmuls + 3 pointwise ops per solve.

from __future__ import annotations

import logging
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import ttnn
    HAS_TTNN = True
except ImportError:
    HAS_TTNN = False


def _build_dct2_matrix(N: int) -> np.ndarray:
    """DCT-II basis matching DREAMPlace's FFT-based implementation.

    DREAMPlace implements the standard DCT-II with (2/N) normalization:
        C[k, n] = (2/N) * cos(pi * k * (n + 0.5) / N)
    where k is the output frequency index (row) and n is the input spatial
    index (column).  The 2D dct2 applies this to both dimensions:
        auv = DCT_M @ rho @ DCT_N^T
    """
    k = np.arange(N, dtype=np.float64).reshape(N, 1)
    n = np.arange(N, dtype=np.float64).reshape(1, N)
    return (2.0 / N) * np.cos(np.pi * k * (n + 0.5) / N)


def _build_idct_matrix(N: int) -> np.ndarray:
    """IDCT (DCT-III x2): IDCT[k,0]=1, IDCT[k,n]=2*cos(pi*n*(k+0.5)/N) for n>=1"""
    k = np.arange(N, dtype=np.float64).reshape(N, 1)
    n = np.arange(N, dtype=np.float64).reshape(1, N)
    mat = 2.0 * np.cos(np.pi * n * (k + 0.5) / N)
    mat[:, 0] = 1.0
    return mat


def _build_idxst_matrix(N: int) -> np.ndarray:
    """IDXST: IDXST[k,n] = sin(pi * n * (k + 0.5) / N)"""
    k = np.arange(N, dtype=np.float64).reshape(N, 1)
    n = np.arange(N, dtype=np.float64).reshape(1, N)
    return np.sin(np.pi * n * (k + 0.5) / N)


class TTNNFieldSolver:
    """Electrostatic field solver running on Tenstorrent hardware.

    Given a density_map (already normalized by 1/(bin_size_x*bin_size_y)),
    computes field_map_x and field_map_y via DCT-as-matmul.

    All fixed matrices (DCT, IDCT, IDXST, eigenvalue weights) are uploaded
    to the TT device once at construction time and reused across all calls.
    """

    def __init__(self, M: int, N: int, bin_size_x: float, bin_size_y: float,
                 device_id: int = 0, tt_dtype=None, device=None):
        assert HAS_TTNN, "ttnn is not available"
        self.M = M
        self.N = N
        if tt_dtype is None:
            tt_dtype = ttnn.float32
        self.tt_dtype = tt_dtype

        if device is not None:
            self.device = device
            self._owns_device = False
            logger.info("TTNNFieldSolver: using existing TT device for %dx%d grid", M, N)
        else:
            logger.info("TTNNFieldSolver: opening TT device %d for %dx%d grid", device_id, M, N)
            self.device = ttnn.open_device(device_id=device_id, l1_small_size=8192)
            self._owns_device = True

        def _to_tt(arr: np.ndarray):
            return ttnn.from_torch(
                torch.tensor(arr.astype(np.float32), dtype=torch.float32),
                dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=self.device,
            )

        # 2D DCT-II: auv = DCT_M @ rho @ DCT_N^T
        self.DCT_M_tt = _to_tt(_build_dct2_matrix(M))
        self.DCT_N_T_tt = _to_tt(_build_dct2_matrix(N).T.copy())

        # IDXST_IDCT for field_x: IDXST_M @ x @ IDCT_N^T
        self.IDXST_M_tt = _to_tt(_build_idxst_matrix(M))
        self.IDCT_N_T_tt = _to_tt(_build_idct_matrix(N).T.copy())

        # IDCT_IDXST for field_y: IDCT_M @ x @ IDXST_N^T
        self.IDCT_M_tt = _to_tt(_build_idct_matrix(M))
        self.IDXST_N_T_tt = _to_tt(_build_idxst_matrix(N).T.copy())

        # Eigenvalue weights — computed with bin aspect ratio, matching
        # ElectricPotential.forward lines 518-534 in electric_potential.py.
        wu = (2.0 * np.pi * np.arange(M, dtype=np.float64) / M).reshape(M, 1)
        wv = (2.0 * np.pi * np.arange(N, dtype=np.float64) / N).reshape(1, N)
        wv *= (bin_size_x / bin_size_y)  # aspect ratio correction

        wu2_plus_wv2 = wu ** 2 + wv ** 2
        wu2_plus_wv2[0, 0] = 1.0
        inv_wu2_plus_wv2 = 1.0 / wu2_plus_wv2
        inv_wu2_plus_wv2[0, 0] = 0.0

        wu_weights = wu * inv_wu2_plus_wv2 * 0.5   # (M, N)
        wv_weights = wv * inv_wu2_plus_wv2 * 0.5   # (M, N)
        self.wu_weights_tt = _to_tt(wu_weights)
        self.wv_weights_tt = _to_tt(wv_weights)

        # Pre-allocate static buffers to avoid GC pressure and spikes
        zero = torch.zeros(M, N, dtype=torch.float32)
        def _alloc_buf():
            return ttnn.from_torch(zero, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        self._temp = _alloc_buf()
        self._auv = _alloc_buf()
        self._fx_auv = _alloc_buf()
        self._fy_auv = _alloc_buf()
        self._temp_x = _alloc_buf()
        self._field_x_raw = _alloc_buf()
        self._field_x_tt = _alloc_buf()
        self._temp_y = _alloc_buf()
        self._field_y_raw = _alloc_buf()
        self._field_y_tt = _alloc_buf()

        self._upload_time = 0.0
        self._compute_time = 0.0
        self._download_time = 0.0
        self._call_count = 0

        # Optional per-call CSV logging (set by external code before first call)
        # Expected: a csv.writer with header [call, upload_ms, compute_ms, download_ms, total_ms]
        self.timing_csv_writer = None

        logger.info("TTNNFieldSolver: ready (%dx%d, dtype=%s)", M, N, tt_dtype)

    def solve_from_device(self, density_map_tt, bin_area: float):
        """Compute field_x, field_y from a density map already on this TT device.

        Avoids upload overhead when density comes from TTNN density scatter
        (e.g. density_map_scatter_ttnn(..., return_on_device=True)).

        Args:
            density_map_tt: (M, N) ttnn tensor on self.device, TILE_LAYOUT, unnormalized (raw area).
            bin_area: bin_size_x * bin_size_y; density is normalized by 1/bin_area on device.

        Returns:
            (field_x, field_y): each (M, N) torch.float32 on CPU.
        """
        M, N = self.M, self.N
        # Normalize on device: rho = density_map_tt * (1/bin_area)
        rho_tt = ttnn.mul(density_map_tt, 1.0 / bin_area)
        ttnn.deallocate(density_map_tt)

        # Same DCT pipeline as solve(), but no upload
        ttnn.matmul(rho_tt, self.DCT_N_T_tt, optional_output_tensor=self._temp)
        ttnn.deallocate(rho_tt)
        ttnn.matmul(self.DCT_M_tt, self._temp, optional_output_tensor=self._auv)

        ttnn.mul(self._auv, self.wu_weights_tt, output_tensor=self._fx_auv)
        ttnn.mul(self._auv, self.wv_weights_tt, output_tensor=self._fy_auv)

        ttnn.matmul(self._fx_auv, self.IDCT_N_T_tt, optional_output_tensor=self._temp_x)
        ttnn.matmul(self.IDXST_M_tt, self._temp_x, optional_output_tensor=self._field_x_raw)
        ttnn.mul(self._field_x_raw, 2.0, output_tensor=self._field_x_tt)

        ttnn.matmul(self._fy_auv, self.IDXST_N_T_tt, optional_output_tensor=self._temp_y)
        ttnn.matmul(self.IDCT_M_tt, self._temp_y, optional_output_tensor=self._field_y_raw)
        ttnn.mul(self._field_y_raw, 2.0, output_tensor=self._field_y_tt)

        ttnn.synchronize_device(self.device)

        field_x = ttnn.to_torch(self._field_x_tt)
        field_y = ttnn.to_torch(self._field_y_tt)
        if field_x.dtype == torch.bfloat16:
            field_x = field_x.float()
        if field_y.dtype == torch.bfloat16:
            field_y = field_y.float()
        field_x = field_x[:M, :N].contiguous()
        field_y = field_y[:M, :N].contiguous()
        return field_x, field_y

    def solve(self, density_map: torch.Tensor):
        """Compute field_x, field_y from a normalized density map.

        Args:
            density_map: (M, N) torch.float32 tensor on CPU,
                         already scaled by 1/(bin_size_x * bin_size_y).

        Returns:
            (field_x, field_y): each (M, N) torch.float32 tensors on CPU.
        """
        M, N = self.M, self.N

        # Upload density_map to TT
        t0 = time.perf_counter()
        rho_tt = ttnn.from_torch(
            density_map.detach().float().contiguous(),
            dtype=self.tt_dtype, layout=ttnn.TILE_LAYOUT, device=self.device,
        )
        ttnn.synchronize_device(self.device)
        t_upload = time.perf_counter() - t0

        # Compute on TT — explicitly reusing pre-allocated static buffers.
        # This completely eliminates device memory allocations inside the hot loop,
        # preventing Python GC synchronization spikes.
        t0 = time.perf_counter()

        # 2D DCT-II: auv = DCT_M @ rho @ DCT_N^T
        ttnn.matmul(rho_tt, self.DCT_N_T_tt, optional_output_tensor=self._temp)
        ttnn.deallocate(rho_tt)
        ttnn.matmul(self.DCT_M_tt, self._temp, optional_output_tensor=self._auv)

        # Eigenvalue scaling
        ttnn.mul(self._auv, self.wu_weights_tt, output_tensor=self._fx_auv)
        ttnn.mul(self._auv, self.wv_weights_tt, output_tensor=self._fy_auv)

        # IDXST_IDCT → field_x: 2 * IDXST_M @ fx_auv @ IDCT_N^T
        ttnn.matmul(self._fx_auv, self.IDCT_N_T_tt, optional_output_tensor=self._temp_x)
        ttnn.matmul(self.IDXST_M_tt, self._temp_x, optional_output_tensor=self._field_x_raw)
        ttnn.mul(self._field_x_raw, 2.0, output_tensor=self._field_x_tt)

        # IDCT_IDXST → field_y: 2 * IDCT_M @ fy_auv @ IDXST_N^T
        ttnn.matmul(self._fy_auv, self.IDXST_N_T_tt, optional_output_tensor=self._temp_y)
        ttnn.matmul(self.IDCT_M_tt, self._temp_y, optional_output_tensor=self._field_y_raw)
        ttnn.mul(self._field_y_raw, 2.0, output_tensor=self._field_y_tt)

        ttnn.synchronize_device(self.device)
        t_compute = time.perf_counter() - t0

        # Download results
        t0 = time.perf_counter()
        field_x = ttnn.to_torch(self._field_x_tt)
        field_y = ttnn.to_torch(self._field_y_tt)
        t_download = time.perf_counter() - t0

        # Cast bfloat16 → float32 if needed
        if field_x.dtype == torch.bfloat16:
            field_x = field_x.float()
        if field_y.dtype == torch.bfloat16:
            field_y = field_y.float()

        # Trim padding from TILE_LAYOUT (multiples of 32)
        field_x = field_x[:M, :N].contiguous()
        field_y = field_y[:M, :N].contiguous()

        self._upload_time += t_upload
        self._compute_time += t_compute
        self._download_time += t_download

        if self.timing_csv_writer is not None:
            self.timing_csv_writer.writerow([
                self._call_count,
                round(t_upload * 1000, 4),
                round(t_compute * 1000, 4),
                round(t_download * 1000, 4),
                round((t_upload + t_compute + t_download) * 1000, 4),
            ])
        self._call_count += 1

        return field_x, field_y

    def reset_timing_stats(self) -> None:
        """Clear upload/compute/download accumulators and call count.

        Call after warmup so :meth:`report_timing` reflects steady-state ``solve()`` calls only.
        """
        self._upload_time = 0.0
        self._compute_time = 0.0
        self._download_time = 0.0
        self._call_count = 0

    def report_timing(self):
        if self._call_count == 0:
            return "TTNNFieldSolver: no calls yet"
        return (
            f"TTNNFieldSolver: {self._call_count} calls, "
            f"avg upload={self._upload_time / self._call_count * 1000:.3f}ms, "
            f"avg compute={self._compute_time / self._call_count * 1000:.3f}ms, "
            f"avg download={self._download_time / self._call_count * 1000:.3f}ms, "
            f"avg total={(self._upload_time + self._compute_time + self._download_time) / self._call_count * 1000:.3f}ms"
        )

    def close(self):
        if hasattr(self, 'device') and self.device is not None:
            logger.info(self.report_timing())
            if getattr(self, "_owns_device", True):
                ttnn.close_device(self.device)
            self.device = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
