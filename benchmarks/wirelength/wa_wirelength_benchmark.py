#!/usr/bin/env python3
"""
Benchmark and correctness test: TTNN vs DREAMPlace CPU for smooth wirelength.

Tests both Weighted-Average (WA) and Log-Sum-Exp (LSE) forward passes at
adaptec1/adaptec3 scale. Correctness is verified against DREAMPlace's compiled
CPU 'merged' ops (if installed) and a pure-PyTorch reference (always available).

Run (self-contained under TTPlace; no scripts outside TTPlace required):
    cd TTPlace/dreamplace_ttnn_profile/scripts
    export PYTHONPATH=/path/to/TTPlace/DREAMPlace:$PYTHONPATH
    python wa_wirelength_cpu_vs_ttnn_benchmark.py

════════════════════════════════════════════════════════════════════
FEASIBILITY SUMMARY  (ref: DREAMPlace/docs/TTNN_OPERATIONS_REFERENCE.md)
════════════════════════════════════════════════════════════════════

FEASIBLE — forward pass only.

  BUILDING BLOCKS USED:
    Element-wise (§5, §6): ttnn.exp, ttnn.log (on CPU after download),
      ttnn.add, ttnn.subtract, ttnn.multiply, ttnn.clip.
    Reductions (§11): ttnn.max, ttnn.min (per-row, for max-shift stability).
    Matrix multiply (§8): ttnn.matmul — used to aggregate per-net sums in a
      single batched operation instead of ragged scatter_add.

  KEY CHALLENGE — PER-NET GROUPED REDUCTION:
    DREAMPlace uses CSR (flat_netpin + netpin_start) for per-net scatter sums.
    TTNN scatter_add has a 256-element row limit (density-map scatter work).
    WORKAROUND (same pattern as ttnn_density_map_scatter_robert.py):
      1. Lay out pins in a DENSE NET-MAJOR matrix: shape (num_nets, max_degree),
         padded to TT_TILE=32 multiples, zeros for unused/masked slots.
      2. ttnn.max / ttnn.min along dim=1 → per-net max/min (numerically stable
         shifted exp).
      3. Element-wise exp + multiply → per-net feature arrays.
      4. ttnn.matmul(features, ones_col) → per-net sums in one GEMM.
    Memory: adaptec1 ≈ 221K nets × 64 pins (padded) × 4 bytes ≈ 56 MB — fits
    in DRAM.

  NUMERICAL SAFETY — MASKED-ENTRY OVERFLOW & TTNN exp() NaN RANGE:
    In the negative-direction exponential dnx = (min_x − x_i)/γ, padded entries
    have x_i = 0 and min_x > 0, giving dnx > 0 → exp overflow.
    FIX: multiply dnx by the validity mask BEFORE exp (zeroes padded entries,
    and all valid entries already satisfy x_i ≥ min_x so dnx ≤ 0).
    For dx = (x_i − max_x)/γ: padded x_i = 0 < max_x, so dx < 0 always — safe
    without masking.
    TTNN BF16 DENORMAL BUG: ttnn.exp() returns NaN for inputs in the range
    approximately [−89, −87]. This corresponds to the float32 normal/denormal
    boundary: exp(−87.34) ≈ FLT_MIN. TTNN's exp kernel (which internally uses
    bfloat16) flushes values below ~−89 to 0 but NaNs those in [−89, −87].
    FIX: clip all exponent arguments to [−85, 0]. exp(−85) ≈ 8.1e-37 is
    negligible in WA/LSE sums and safely above the denormal boundary.

  DTYPE: float32 on device (not bfloat16). The log() step after accumulating
    exp sums loses meaningful precision in bf16 for small sums. float32 is
    listed in TTNN_OPERATIONS_REFERENCE.md §2.

  NOT IN SCOPE HERE — backward pass:
    Per-pin gradient scatter from per-net terms requires scatter_add at scale,
    which has the 256-row limit. For placement integration, backward can stay on
    CPU via DREAMPlace's compiled autograd; TTNN handles the forward only.

════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

# ── TTNN import (optional, skipped gracefully if no device) ─────────────────
try:
    import ttnn
    HAS_TTNN = True
except ImportError:
    HAS_TTNN = False

# ── DREAMPlace CPU ops (optional, needs compiled extensions on PYTHONPATH) ───
HAS_DP_WA = HAS_DP_LSE = False
_DP_WA = _DP_LSE = None
try:
    # This file lives in TTPort/benchmarks/wirelength/ — dreamplace_ref is TTPort/dreamplace_ref.
    _here = os.path.dirname(os.path.abspath(__file__))
    _ttport = os.path.abspath(os.path.join(_here, "..", ".."))
    _dp_root = os.path.join(_ttport, "dreamplace_ref")
    if _dp_root not in sys.path:
        sys.path.insert(0, _dp_root)
    from dreamplace.ops.weighted_average_wirelength.weighted_average_wirelength import (
        WeightedAverageWirelength as _DP_WA,
    )
    HAS_DP_WA = True
except Exception:
    pass

try:
    from dreamplace.ops.logsumexp_wirelength.logsumexp_wirelength import (
        LogSumExpWirelength as _DP_LSE,
    )
    HAS_DP_LSE = True
except Exception:
    pass

# ── Constants ────────────────────────────────────────────────────────────────
TT_TILE = 32

# Masking sentinel for algebraic max/min masking.
# Must satisfy: BIG > layout_wh and BIG * layout_wh representable in float32.
# With layout_wh=6000: BIG=1e4 keeps (x ± BIG) exact in float32.
BIG = 1e4


def pad_to(n: int, tile: int = TT_TILE) -> int:
    return ((n + tile - 1) // tile) * tile


# ════════════════════════════════════════════════════════════════════════════
# Synthetic netlist generation
# ════════════════════════════════════════════════════════════════════════════

def make_netlist(
    num_nets: int,
    layout_wh: float = 6000.0,
    max_net_degree: int = 100,
    seed: int = 42,
) -> dict:
    """
    Synthetic netlist with a degree distribution matching ISPD2005 benchmarks:
    50% degree-2, 25% degree-3, 15% degree-4, 7% degree 5-10, 3% degree 11-50.

    Returns both flat CSR (for DREAMPlace ops) and dense net-major (for TTNN).
    Nets with degree > max_net_degree are masked out (mirroring DREAMPlace's
    net_mask_ignore_large_degrees).
    """
    rng = np.random.default_rng(seed)
    probs = [0.50, 0.25, 0.15, 0.07, 0.03]
    ranges = [(2, 2), (3, 3), (4, 4), (5, 10), (11, 50)]
    choice = rng.choice(len(probs), size=num_nets, p=probs)
    degrees = np.array(
        [rng.integers(lo, hi + 1) for lo, hi in [ranges[c] for c in choice]],
        dtype=np.int32,
    )

    net_mask = (degrees <= max_net_degree).astype(np.uint8)
    num_pins = int(degrees.sum())

    # CSR connectivity
    netpin_start = np.zeros(num_nets + 1, dtype=np.int32)
    np.cumsum(degrees, out=netpin_start[1:])
    flat_netpin = np.arange(num_pins, dtype=np.int32)  # identity ordering
    pin2net = np.zeros(num_pins, dtype=np.int32)
    for n in range(num_nets):
        pin2net[netpin_start[n] : netpin_start[n + 1]] = n

    # Pin coordinates uniform in [0, layout_wh] (typical global-placement scale)
    pin_x = rng.uniform(0, layout_wh, num_pins).astype(np.float32)
    pin_y = rng.uniform(0, layout_wh, num_pins).astype(np.float32)
    pos = np.concatenate([pin_x, pin_y])  # DREAMPlace layout: [x..., y...]

    net_weights = np.ones(num_nets, dtype=np.float32)
    pin_mask = np.zeros(num_pins, dtype=np.uint8)  # no pins masked for gradient

    # Dense net-major layout (num_nets × max_degree), padded to TT_TILE
    valid_degs = degrees[net_mask == 1]
    max_deg = int(valid_degs.max()) if len(valid_degs) > 0 else 2
    nc_p = pad_to(max_deg)
    nr_p = pad_to(num_nets)

    x_nm = np.zeros((nr_p, nc_p), dtype=np.float32)
    y_nm = np.zeros((nr_p, nc_p), dtype=np.float32)
    mask_nm = np.zeros((nr_p, nc_p), dtype=np.float32)

    for n in range(num_nets):
        if not net_mask[n]:
            continue
        s, e = int(netpin_start[n]), int(netpin_start[n + 1])
        d = e - s
        pins = flat_netpin[s:e]
        x_nm[n, :d] = pin_x[pins]
        y_nm[n, :d] = pin_y[pins]
        mask_nm[n, :d] = 1.0

    return {
        # Flat / CSR (DREAMPlace + scatter-based PyTorch reference)
        "pos": torch.from_numpy(pos.astype(np.float32)),
        "flat_netpin": torch.from_numpy(flat_netpin),
        "netpin_start": torch.from_numpy(netpin_start),
        "pin2net": torch.from_numpy(pin2net),
        "net_weights": torch.from_numpy(net_weights),
        "net_mask": torch.from_numpy(net_mask),
        "pin_mask": torch.from_numpy(pin_mask),
        "num_pins": num_pins,
        "num_nets": num_nets,
        # Dense / net-major (TTNN + PyTorch reference using same layout)
        "x_nm": torch.from_numpy(x_nm),
        "y_nm": torch.from_numpy(y_nm),
        "mask_nm": torch.from_numpy(mask_nm),
        "max_deg": max_deg,
        "nr_p": nr_p,
        "nc_p": nc_p,
    }


# ════════════════════════════════════════════════════════════════════════════
# Pure-PyTorch CPU reference  (always available, uses same dense layout as TTNN)
# ════════════════════════════════════════════════════════════════════════════

def pytorch_wa(
    x_nm: torch.Tensor,
    y_nm: torch.Tensor,
    mask_nm: torch.Tensor,
    net_weights: torch.Tensor,
    net_mask: torch.Tensor,
    gamma: float,
    num_nets: int,
) -> torch.Tensor:
    """
    Weighted-average wirelength (dense net-major layout, pure PyTorch).

    WA_x_k = (Σ x_i · exp((x_i − max_x_k)/γ)) / (Σ exp((x_i − max_x_k)/γ))
            − (Σ x_i · exp((min_x_k − x_i)/γ)) / (Σ exp((min_x_k − x_i)/γ))

    Both halves are numerically stable (exponents ≤ 0 for valid pins).
    Masked (padded) entries are neutralised before exp to prevent overflow.
    """
    inv_g = 1.0 / gamma
    m = mask_nm[:num_nets].float()   # (num_nets, nc_p)
    x = x_nm[:num_nets].float()
    y = y_nm[:num_nets].float()

    # Masked max/min via sentinel values so invalid slots don't affect result.
    x_mx = (x * m + (-BIG) * (1.0 - m)).max(dim=1).values  # (num_nets,)
    x_mn = (x * m +   BIG  * (1.0 - m)).min(dim=1).values
    y_mx = (y * m + (-BIG) * (1.0 - m)).max(dim=1).values
    y_mn = (y * m +   BIG  * (1.0 - m)).min(dim=1).values

    # dx = (x_i − max_x_k)/γ ≤ 0 for valid pins; for padded x_i=0 < max_x → OK.
    dx  = (x - x_mx.unsqueeze(1)) * inv_g
    # dnx = (min_x_k − x_i)/γ ≤ 0 for valid; padded x_i=0 gives positive → must mask.
    dnx = (x_mn.unsqueeze(1) - x) * inv_g * m
    dy  = (y - y_mx.unsqueeze(1)) * inv_g
    dny = (y_mn.unsqueeze(1) - y) * inv_g * m

    ex   = torch.exp(dx)  * m
    enx  = torch.exp(dnx) * m
    ey   = torch.exp(dy)  * m
    eny  = torch.exp(dny) * m

    eps = 1e-15
    sum_ex   = ex.sum(1);  sum_xex  = (x * ex).sum(1)
    sum_enx  = enx.sum(1); sum_xenx = (x * enx).sum(1)
    sum_ey   = ey.sum(1);  sum_yey  = (y * ey).sum(1)
    sum_eny  = eny.sum(1); sum_yeny = (y * eny).sum(1)

    wa_x = sum_xex / (sum_ex + eps) - sum_xenx / (sum_enx + eps)
    wa_y = sum_yey / (sum_ey + eps) - sum_yeny / (sum_eny + eps)

    nw = net_weights[:num_nets] * net_mask[:num_nets].float()
    return (nw * (wa_x + wa_y)).sum()


def pytorch_lse(
    x_nm: torch.Tensor,
    y_nm: torch.Tensor,
    mask_nm: torch.Tensor,
    net_weights: torch.Tensor,
    net_mask: torch.Tensor,
    gamma: float,
    num_nets: int,
) -> torch.Tensor:
    """
    Log-sum-exp wirelength (dense net-major layout, pure PyTorch).

    LSE_x_k = γ·log(Σ e^{x/γ}) + γ·log(Σ e^{-x/γ})
             = (max_x_k − min_x_k) + γ·(log(sum_ex_k) + log(sum_enx_k))

    where sum_ex_k = Σ exp((x_i − max_x_k)/γ)  [max-shifted for stability]
          sum_enx_k= Σ exp((min_x_k − x_i)/γ)  [min-shifted for stability]
    """
    inv_g = 1.0 / gamma
    m = mask_nm[:num_nets].float()
    x = x_nm[:num_nets].float()
    y = y_nm[:num_nets].float()

    x_mx = (x * m + (-BIG) * (1.0 - m)).max(dim=1).values
    x_mn = (x * m +   BIG  * (1.0 - m)).min(dim=1).values
    y_mx = (y * m + (-BIG) * (1.0 - m)).max(dim=1).values
    y_mn = (y * m +   BIG  * (1.0 - m)).min(dim=1).values

    dx  = (x - x_mx.unsqueeze(1)) * inv_g          # ≤ 0 for valid, < 0 for padded → OK
    dnx = (x_mn.unsqueeze(1) - x) * inv_g * m      # masked before exp
    dy  = (y - y_mx.unsqueeze(1)) * inv_g
    dny = (y_mn.unsqueeze(1) - y) * inv_g * m

    sum_ex  = (torch.exp(dx)  * m).sum(1)
    sum_enx = (torch.exp(dnx) * m).sum(1)
    sum_ey  = (torch.exp(dy)  * m).sum(1)
    sum_eny = (torch.exp(dny) * m).sum(1)

    eps = 1e-15
    lse_x = (x_mx - x_mn) + gamma * (torch.log(sum_ex + eps) + torch.log(sum_enx + eps))
    lse_y = (y_mx - y_mn) + gamma * (torch.log(sum_ey + eps) + torch.log(sum_eny + eps))

    nw = net_weights[:num_nets] * net_mask[:num_nets].float()
    return (nw * (lse_x + lse_y)).sum()


# ════════════════════════════════════════════════════════════════════════════
# DREAMPlace CPU op reference  (needs compiled extensions)
# ════════════════════════════════════════════════════════════════════════════

def dp_wa_ref(data: dict, gamma: float) -> float | None:
    """Run DREAMPlace's compiled CPU 'merged' WA op. Returns scalar or None."""
    if not HAS_DP_WA:
        return None
    gamma_t = torch.tensor(gamma)
    op = _DP_WA(
        flat_netpin=data["flat_netpin"],
        netpin_start=data["netpin_start"],
        pin2net_map=data["pin2net"],
        net_weights=data["net_weights"],
        net_mask=data["net_mask"],
        pin_mask=data["pin_mask"],
        gamma=gamma_t,
        algorithm="merged",
    )
    return op(data["pos"]).item()


def dp_lse_ref(data: dict, gamma: float) -> float | None:
    """Run DREAMPlace's compiled CPU 'merged' LSE op. Returns scalar or None."""
    if not HAS_DP_LSE:
        return None
    gamma_t = torch.tensor(gamma)
    try:
        op = _DP_LSE(
            flat_netpin=data["flat_netpin"],
            netpin_start=data["netpin_start"],
            pin2net_map=data["pin2net"],
            net_weights=data["net_weights"],
            net_mask=data["net_mask"],
            pin_mask=data["pin_mask"],
            gamma=gamma_t,
            algorithm="merged",
        )
        return op(data["pos"]).item()
    except Exception as exc:
        print(f"    [WARN] DREAMPlace LSE ref failed: {exc}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# TTNN implementation
# ════════════════════════════════════════════════════════════════════════════

def _up(t: torch.Tensor, dev) -> "ttnn.Tensor":
    """Upload a float32 CPU tensor to device in TILE_LAYOUT."""
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)


def _masked_max_min(xt, yt, mt, nr_p: int):
    """
    Compute per-net (row) max/min for x and y on device.

    Max direction — use x·m directly:
      Masked entries contribute 0.  Since all pin coordinates ≥ 0, the
      per-row max is still the actual max of valid pins (0 ≤ valid coords).
      Critically: for ALL-zero-mask rows (filtered/padded nets) this gives
      xmx = 0, so dx = (0 − 0)/γ = 0, exp(0) = 1, and *m = 0 (safe).
      The old (x+BIG)·m − BIG formula gave xmx = −BIG for those rows,
      causing dx = BIG/γ ≈ +667 → exp overflow → NaN.

    Min direction — BIG sentinel:
      (x − BIG)·m + BIG: valid → x, masked → +BIG.
      For all-zero-mask rows: xmn = BIG, but dnx is multiplied by mt before
      exp, so dnx = (BIG − 0)/γ * 0 = 0 → safe.

    Returns (xmx, xmn, ymx, ymn) each shape (nr_p, 1).
    """
    # Max: x*m — masked entries are 0, valid entries are x ≥ 0.
    x_for_max = ttnn.multiply(xt, mt)
    x_for_min = ttnn.add(ttnn.multiply(ttnn.subtract(xt, BIG), mt), BIG)
    y_for_max = ttnn.multiply(yt, mt)
    y_for_min = ttnn.add(ttnn.multiply(ttnn.subtract(yt, BIG), mt), BIG)

    xmx = ttnn.max(x_for_max, dim=1, keepdim=True)  # (nr_p, 1)
    xmn = ttnn.min(x_for_min, dim=1, keepdim=True)
    ymx = ttnn.max(y_for_max, dim=1, keepdim=True)
    ymn = ttnn.min(y_for_min, dim=1, keepdim=True)

    for t in [x_for_max, x_for_min, y_for_max, y_for_min]:
        ttnn.deallocate(t)

    return xmx, xmn, ymx, ymn


def ttnn_wa(
    x_nm: torch.Tensor,
    y_nm: torch.Tensor,
    mask_nm: torch.Tensor,
    net_weights: torch.Tensor,
    net_mask: torch.Tensor,
    gamma: float,
    num_nets: int,
    nc_p: int,
    dev,
    ones_tt,  # precomputed static GEMM kernel (8*nc_p_padded × 8 on device)
) -> tuple[float, dict]:
    """
    Weighted-average wirelength forward on TTNN.

    Algorithm:
      1. Upload x_nm, y_nm, mask_nm                            [3 uploads]
      2. Per-net max/min via masked algebraic reduction
      3. Compute exp terms (masked before exp to prevent overflow)
      4. GEMM: concat [ex, x·ex, enx, x·enx, ey, y·ey, eny, y·eny] → (nr_p, 8·nc_p)
               matmul with ones_col (8·nc_p × 8) → per-net 8 sums (nr_p, 8)
      5. Download sums; final WA scalar on CPU                  [tiny: num_nets ops]

    Returns (scalar_wl, timing_dict).
    """
    inv_g = 1.0 / gamma
    timing: dict[str, float] = {}

    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    xt = _up(x_nm, dev)
    yt = _up(y_nm, dev)
    mt = _up(mask_nm, dev)
    ttnn.synchronize_device(dev)
    timing["upload"] = time.perf_counter() - t0

    t1 = time.perf_counter()

    xmx, xmn, ymx, ymn = _masked_max_min(xt, yt, mt, x_nm.shape[0])

    # dx = (x_i − xmx)/γ.  xmx = max(x·m), so for valid pins dx ≤ 0.
    # For all-zero-mask rows: xmx = 0, x_i = 0 → dx = 0 → exp(0) = 1, *mt = 0.
    # CLAMP to [−85, 0]:
    #   Upper bound 0: corrects bfloat16 rounding of xmx that can make dx slightly
    #                  positive for the near-max pin (exp(small_positive) → 1).
    #   Lower bound −85: avoids TTNN bfloat16 denormal NaN range (~[−89, −87]).
    #                  exp(−85) ≈ 8.1e-37 is negligible for WA/LSE sums.
    dx  = ttnn.clip(ttnn.multiply(ttnn.subtract(xt, xmx), inv_g), -85.0, 0.0)
    # dnx = (xmn − x_i)/γ * m.  Masked before exp: all-zero rows get 0.
    # Valid pins: x_i ≥ xmn → dnx ≤ 0.  Wide nets can give dnx ≪ −85 → NaN without clip.
    dnx = ttnn.clip(ttnn.multiply(ttnn.multiply(ttnn.subtract(xmn, xt), inv_g), mt), -85.0, 0.0)
    dy  = ttnn.clip(ttnn.multiply(ttnn.subtract(yt, ymx), inv_g), -85.0, 0.0)
    dny = ttnn.clip(ttnn.multiply(ttnn.multiply(ttnn.subtract(ymn, yt), inv_g), mt), -85.0, 0.0)

    for t in [xmx, xmn, ymx, ymn]:
        ttnn.deallocate(t)

    ex   = ttnn.multiply(ttnn.exp(dx),  mt);  ttnn.deallocate(dx)
    enx  = ttnn.multiply(ttnn.exp(dnx), mt);  ttnn.deallocate(dnx)
    ey   = ttnn.multiply(ttnn.exp(dy),  mt);  ttnn.deallocate(dy)
    eny  = ttnn.multiply(ttnn.exp(dny), mt);  ttnn.deallocate(dny)
    ttnn.deallocate(mt)

    xex  = ttnn.multiply(xt, ex)
    xenx = ttnn.multiply(xt, enx)
    yey  = ttnn.multiply(yt, ey)
    yeny = ttnn.multiply(yt, eny)
    ttnn.deallocate(xt);  ttnn.deallocate(yt)

    # Concatenate 8 feature planes → (nr_p, 8·nc_p), reduce via GEMM → (nr_p, 8)
    features = ttnn.concat([ex, xex, enx, xenx, ey, yey, eny, yeny], dim=1)
    for t in [ex, xex, enx, xenx, ey, yey, eny, yeny]:
        ttnn.deallocate(t)

    sums_tt = ttnn.matmul(features, ones_tt)
    ttnn.deallocate(features)

    ttnn.synchronize_device(dev)
    timing["compute"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    # Shape: (nr_p, pad_to(8)=32); first 8 columns are the real sums.
    sums = ttnn.to_torch(sums_tt).float()[:num_nets, :8]
    ttnn.synchronize_device(dev)
    timing["download"] = time.perf_counter() - t2
    ttnn.deallocate(sums_tt)

    # Final WA aggregation on CPU (num_nets ops — negligible).
    eps = 1e-15
    sum_ex,  sum_xex  = sums[:, 0], sums[:, 1]
    sum_enx, sum_xenx = sums[:, 2], sums[:, 3]
    sum_ey,  sum_yey  = sums[:, 4], sums[:, 5]
    sum_eny, sum_yeny = sums[:, 6], sums[:, 7]

    wa_x = sum_xex / (sum_ex + eps) - sum_xenx / (sum_enx + eps)
    wa_y = sum_yey / (sum_ey + eps) - sum_yeny / (sum_eny + eps)

    nw = net_weights[:num_nets] * net_mask[:num_nets].float()
    wl = (nw * (wa_x + wa_y)).sum().item()
    return wl, timing


def ttnn_lse(
    x_nm: torch.Tensor,
    y_nm: torch.Tensor,
    mask_nm: torch.Tensor,
    net_weights: torch.Tensor,
    net_mask: torch.Tensor,
    gamma: float,
    num_nets: int,
    nc_p: int,
    dev,
    ones_tt,  # precomputed static GEMM kernel (4*nc_p_padded × 4 on device)
) -> tuple[float, dict]:
    """
    Log-sum-exp wirelength forward on TTNN.

    LSE_x_k = (max_x_k − min_x_k) + γ·(log(sum_ex_k) + log(sum_enx_k))

    where sum_ex_k and sum_enx_k are the max-shifted per-net exp sums (same
    exponentials as WA but without the x-weighted numerator terms).
    Only 4 sums needed (vs 8 for WA), so the GEMM is smaller.
    log() is applied on CPU after downloading sums (avoids on-device log).

    Returns (scalar_wl, timing_dict).
    """
    inv_g = 1.0 / gamma
    timing: dict[str, float] = {}

    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    xt = _up(x_nm, dev)
    yt = _up(y_nm, dev)
    mt = _up(mask_nm, dev)
    ttnn.synchronize_device(dev)
    timing["upload"] = time.perf_counter() - t0

    t1 = time.perf_counter()

    xmx, xmn, ymx, ymn = _masked_max_min(xt, yt, mt, x_nm.shape[0])

    # Same exp terms as WA but only denominator sums needed.  Same clip bounds.
    dx  = ttnn.clip(ttnn.multiply(ttnn.subtract(xt, xmx), inv_g), -85.0, 0.0)
    dnx = ttnn.clip(ttnn.multiply(ttnn.multiply(ttnn.subtract(xmn, xt), inv_g), mt), -85.0, 0.0)
    dy  = ttnn.clip(ttnn.multiply(ttnn.subtract(yt, ymx), inv_g), -85.0, 0.0)
    dny = ttnn.clip(ttnn.multiply(ttnn.multiply(ttnn.subtract(ymn, yt), inv_g), mt), -85.0, 0.0)

    # Download max/min for LSE recovery (small: nr_p scalars → CPU log after).
    xmx_cpu = ttnn.to_torch(xmx).float().reshape(-1)[:num_nets]
    xmn_cpu = ttnn.to_torch(xmn).float().reshape(-1)[:num_nets]
    ymx_cpu = ttnn.to_torch(ymx).float().reshape(-1)[:num_nets]
    ymn_cpu = ttnn.to_torch(ymn).float().reshape(-1)[:num_nets]
    for t in [xmx, xmn, ymx, ymn]:
        ttnn.deallocate(t)
    ttnn.deallocate(xt);  ttnn.deallocate(yt)

    ex  = ttnn.multiply(ttnn.exp(dx),  mt);  ttnn.deallocate(dx)
    enx = ttnn.multiply(ttnn.exp(dnx), mt);  ttnn.deallocate(dnx)
    ey  = ttnn.multiply(ttnn.exp(dy),  mt);  ttnn.deallocate(dy)
    eny = ttnn.multiply(ttnn.exp(dny), mt);  ttnn.deallocate(dny)
    ttnn.deallocate(mt)

    # Concatenate 4 feature planes → (nr_p, 4·nc_p), GEMM → (nr_p, 4) sums
    features = ttnn.concat([ex, enx, ey, eny], dim=1)
    for t in [ex, enx, ey, eny]:
        ttnn.deallocate(t)

    sums_tt = ttnn.matmul(features, ones_tt)
    ttnn.deallocate(features)

    ttnn.synchronize_device(dev)
    timing["compute"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    sums = ttnn.to_torch(sums_tt).float()[:num_nets, :4]
    ttnn.synchronize_device(dev)
    timing["download"] = time.perf_counter() - t2
    ttnn.deallocate(sums_tt)

    # LSE recovery on CPU:
    #   γ·log(Σ e^{x/γ}) = max_x + γ·log(sum_ex)
    #   γ·log(Σ e^{-x/γ}) = −min_x + γ·log(sum_enx)
    eps = 1e-15
    sum_ex  = sums[:, 0];  sum_enx = sums[:, 1]
    sum_ey  = sums[:, 2];  sum_eny  = sums[:, 3]

    lse_x = (xmx_cpu - xmn_cpu) + gamma * (torch.log(sum_ex + eps) + torch.log(sum_enx + eps))
    lse_y = (ymx_cpu - ymn_cpu) + gamma * (torch.log(sum_ey + eps) + torch.log(sum_eny + eps))

    nw = net_weights[:num_nets] * net_mask[:num_nets].float()
    wl = (nw * (lse_x + lse_y)).sum().item()
    return wl, timing


def _masked_max_min_2d(xyt, m2t, nr2_p: int):
    """
    Masked max/min for a (2*nr_p, nc_p) stacked x+y tensor.
    Returns (xymx, xymn) each shape (2*nr_p, 1).
    """
    for_max = ttnn.multiply(xyt, m2t)
    for_min = ttnn.add(ttnn.multiply(ttnn.subtract(xyt, BIG), m2t), BIG)
    mx = ttnn.max(for_max, dim=1, keepdim=True)
    mn = ttnn.min(for_min, dim=1, keepdim=True)
    ttnn.deallocate(for_max);  ttnn.deallocate(for_min)
    return mx, mn


def ttnn_wa_opt(
    x_nm: torch.Tensor,
    y_nm: torch.Tensor,
    mask_nm: torch.Tensor,
    net_weights: torch.Tensor,
    net_mask: torch.Tensor,
    gamma: float,
    num_nets: int,
    dev,
) -> tuple[float, dict]:
    """
    Optimised Weighted-Average wirelength on TTNN.

    Three improvements over the GEMM baseline:

    1. x and y are stacked along dim=0 into a single (2·nr_p, nc_p) tensor.
       The entire exp pipeline runs ONCE instead of twice, halving kernel
       dispatch count.

    2. Per-net reductions use ttnn.sum(dim=1) instead of a block-diagonal GEMM.
       The GEMM did (N, 8·nc_p) × (8·nc_p, 32) = 32× more tile multiplications
       than necessary (block-diagonal ones matrix). Direct row-sums skip the
       weight matrix entirely.

    3. Only 4 ttnn.sum calls (on (2N, nc_p) tensors) instead of 8 GEMMs on
       (N, 8·nc_p) tensors, then split x/y on CPU.
    """
    inv_g = 1.0 / gamma
    nr_p  = x_nm.shape[0]   # padded net rows for x (same for y)
    timing: dict[str, float] = {}

    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    xt = _up(x_nm,    dev)
    yt = _up(y_nm,    dev)
    mt = _up(mask_nm, dev)
    # Stack x and y: [x_rows ; y_rows] along dim=0 → (2·nr_p, nc_p)
    xyt = ttnn.concat([xt, yt], dim=0);   ttnn.deallocate(xt);  ttnn.deallocate(yt)
    m2t = ttnn.concat([mt, mt], dim=0);   ttnn.deallocate(mt)
    ttnn.synchronize_device(dev)
    timing["upload"] = time.perf_counter() - t0

    t1 = time.perf_counter()

    # Per-net max/min on (2·nr_p, nc_p): one pass covers both x and y.
    xymx, xymn = _masked_max_min_2d(xyt, m2t, 2 * nr_p)

    # Exponent args, clipped to [−85, 0] to avoid TTNN bfloat16 denormal NaN.
    d_fwd = ttnn.clip(ttnn.multiply(ttnn.subtract(xyt, xymx), inv_g), -85.0, 0.0)
    d_neg = ttnn.clip(ttnn.multiply(ttnn.multiply(
        ttnn.subtract(xymn, xyt), inv_g), m2t), -85.0, 0.0)
    ttnn.deallocate(xymx);  ttnn.deallocate(xymn)

    # exp, masked  →  (2·nr_p, nc_p) holding [ex; ey] and [enx; eny]
    e_fwd = ttnn.multiply(ttnn.exp(d_fwd), m2t);  ttnn.deallocate(d_fwd)
    e_neg = ttnn.multiply(ttnn.exp(d_neg), m2t);  ttnn.deallocate(d_neg)
    ttnn.deallocate(m2t)

    # Weighted numerators: coord × exp  →  [x·ex; y·ey] and [x·enx; y·eny]
    xye_fwd = ttnn.multiply(xyt, e_fwd)
    xye_neg = ttnn.multiply(xyt, e_neg)
    ttnn.deallocate(xyt)

    # Row sums: 4 ttnn.sum calls (each on (2·nr_p, nc_p)) → (2·nr_p, 1)
    s_ef   = ttnn.sum(e_fwd,   dim=1, keepdim=True);  ttnn.deallocate(e_fwd)
    s_en   = ttnn.sum(e_neg,   dim=1, keepdim=True);  ttnn.deallocate(e_neg)
    s_xef  = ttnn.sum(xye_fwd, dim=1, keepdim=True);  ttnn.deallocate(xye_fwd)
    s_xen  = ttnn.sum(xye_neg, dim=1, keepdim=True);  ttnn.deallocate(xye_neg)

    ttnn.synchronize_device(dev)
    timing["compute"] = time.perf_counter() - t1

    # Download: convert TILE→ROW_MAJOR before to_torch so we transfer (2N,1)
    # exact bytes instead of (2N,32) tile-padded = 32× less PCIe traffic.
    t2 = time.perf_counter()
    def _dl(t):
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        cpu = ttnn.to_torch(rm).float().reshape(-1)
        ttnn.deallocate(rm)
        return cpu

    ef  = _dl(s_ef);   ttnn.deallocate(s_ef)
    en  = _dl(s_en);   ttnn.deallocate(s_en)
    xef = _dl(s_xef);  ttnn.deallocate(s_xef)
    xen = _dl(s_xen);  ttnn.deallocate(s_xen)
    ttnn.synchronize_device(dev)
    timing["download"] = time.perf_counter() - t2

    nr = num_nets
    sum_ex,  sum_ey   = ef [:nr],  ef [nr:2*nr]
    sum_enx, sum_eny  = en [:nr],  en [nr:2*nr]
    sum_xex, sum_yey  = xef[:nr],  xef[nr:2*nr]
    sum_xenx,sum_yeny = xen[:nr],  xen[nr:2*nr]

    eps = 1e-15
    wa_x = sum_xex  / (sum_ex  + eps) - sum_xenx / (sum_enx + eps)
    wa_y = sum_yey  / (sum_ey  + eps) - sum_yeny / (sum_eny + eps)
    nw = net_weights[:nr] * net_mask[:nr].float()
    wl = (nw * (wa_x + wa_y)).sum().item()
    return wl, timing


def ttnn_lse_opt(
    x_nm: torch.Tensor,
    y_nm: torch.Tensor,
    mask_nm: torch.Tensor,
    net_weights: torch.Tensor,
    net_mask: torch.Tensor,
    gamma: float,
    num_nets: int,
    dev,
) -> tuple[float, dict]:
    """
    Optimised Log-Sum-Exp wirelength on TTNN.

    Improvements over GEMM baseline:
    1. x+y stacked into (2N, nc_p) — single pipeline pass for both dimensions.
    2. ttnn.sum(dim=1) instead of block-diagonal GEMM.
    3. xymx/xymn stay on device until the final download batch — no mid-pipeline
       sync barriers (the original LSE had 4 to_torch stalls before exp).
    """
    inv_g = 1.0 / gamma
    nr_p  = x_nm.shape[0]
    timing: dict[str, float] = {}

    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    xt = _up(x_nm,    dev)
    yt = _up(y_nm,    dev)
    mt = _up(mask_nm, dev)
    xyt = ttnn.concat([xt, yt], dim=0);  ttnn.deallocate(xt);  ttnn.deallocate(yt)
    m2t = ttnn.concat([mt, mt], dim=0);  ttnn.deallocate(mt)
    ttnn.synchronize_device(dev)
    timing["upload"] = time.perf_counter() - t0

    t1 = time.perf_counter()

    # Max/min stay on device — NOT downloaded here (unlike original LSE).
    xymx, xymn = _masked_max_min_2d(xyt, m2t, 2 * nr_p)

    d_fwd = ttnn.clip(ttnn.multiply(ttnn.subtract(xyt, xymx), inv_g), -85.0, 0.0)
    d_neg = ttnn.clip(ttnn.multiply(ttnn.multiply(
        ttnn.subtract(xymn, xyt), inv_g), m2t), -85.0, 0.0)
    ttnn.deallocate(xyt)

    e_fwd = ttnn.multiply(ttnn.exp(d_fwd), m2t);  ttnn.deallocate(d_fwd)
    e_neg = ttnn.multiply(ttnn.exp(d_neg), m2t);  ttnn.deallocate(d_neg)
    ttnn.deallocate(m2t)

    s_ef = ttnn.sum(e_fwd, dim=1, keepdim=True);  ttnn.deallocate(e_fwd)
    s_en = ttnn.sum(e_neg, dim=1, keepdim=True);  ttnn.deallocate(e_neg)
    # xymx and xymn are still on device here — no stall introduced.

    ttnn.synchronize_device(dev)
    timing["compute"] = time.perf_counter() - t1

    # Download via ROW_MAJOR to avoid tile-padding waste ((2N,1) → (2N,32)).
    t2 = time.perf_counter()
    def _dl(t):
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        cpu = ttnn.to_torch(rm).float().reshape(-1)
        ttnn.deallocate(rm)
        return cpu

    ef = _dl(s_ef);   ttnn.deallocate(s_ef)
    en = _dl(s_en);   ttnn.deallocate(s_en)
    mx = _dl(xymx);   ttnn.deallocate(xymx)
    mn = _dl(xymn);   ttnn.deallocate(xymn)
    ttnn.synchronize_device(dev)
    timing["download"] = time.perf_counter() - t2

    nr = num_nets
    sum_ex,  sum_ey   = ef[:nr],  ef[nr:2*nr]
    sum_enx, sum_eny  = en[:nr],  en[nr:2*nr]
    xmx_cpu, ymx_cpu  = mx[:nr],  mx[nr:2*nr]
    xmn_cpu, ymn_cpu  = mn[:nr],  mn[nr:2*nr]

    eps = 1e-15
    lse_x = (xmx_cpu - xmn_cpu) + gamma * (
        torch.log(sum_ex  + eps) + torch.log(sum_enx + eps))
    lse_y = (ymx_cpu - ymn_cpu) + gamma * (
        torch.log(sum_ey  + eps) + torch.log(sum_eny + eps))

    nw = net_weights[:nr] * net_mask[:nr].float()
    wl = (nw * (lse_x + lse_y)).sum().item()
    return wl, timing


def _make_ones_tt(nc_p: int, n_sums: int, dev) -> "ttnn.Tensor":
    """
    Build the static GEMM reduction kernel for n_sums aggregations.

    Layout: (n_sums·nc_p padded, n_sums) where block i has a column of 1s
    in column i for rows [i·nc_p, (i+1)·nc_p).  After matmul with a feature
    tensor (nr_p, n_sums·nc_p), each output column i is the sum of feature
    plane i across the nc_p pin columns — i.e. one scalar per net.
    """
    n_total = n_sums * nc_p
    n_total_p = pad_to(n_total)
    block = torch.zeros(n_total_p, n_sums, dtype=torch.float32)
    for i in range(n_sums):
        block[i * nc_p : (i + 1) * nc_p, i] = 1.0
    return _up(block, dev)


# ════════════════════════════════════════════════════════════════════════════
# Timing helpers
# ════════════════════════════════════════════════════════════════════════════

def _time_cpu_fn(fn, warmup: int = 2, runs: int = 5) -> tuple[float, float]:
    """Returns (result, median_ms)."""
    result = None
    times = []
    for i in range(warmup + runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return float(result), 1000.0 * float(np.median(times[warmup:]))


# ════════════════════════════════════════════════════════════════════════════
# Per-scale benchmark
# ════════════════════════════════════════════════════════════════════════════

def run_benchmark(num_nets: int, label: str, gamma: float, runs: int = 5, warmup: int = 2):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {label}  |  num_nets={num_nets:,}  |  gamma={gamma}")
    print(sep)

    print("  Generating netlist...", end=" ", flush=True)
    t0 = time.perf_counter()
    data = make_netlist(num_nets)
    print(
        f"done ({time.perf_counter() - t0:.1f}s)  "
        f"pins={data['num_pins']:,}  max_deg={data['max_deg']}  "
        f"dense=({data['nr_p']}×{data['nc_p']})"
    )

    # ── PyTorch reference ────────────────────────────────────────────────────
    def _pt_wa():
        return pytorch_wa(
            data["x_nm"], data["y_nm"], data["mask_nm"],
            data["net_weights"], data["net_mask"], gamma, data["num_nets"]
        )

    def _pt_lse():
        return pytorch_lse(
            data["x_nm"], data["y_nm"], data["mask_nm"],
            data["net_weights"], data["net_mask"], gamma, data["num_nets"]
        )

    ref_wa,  pt_wa_ms  = _time_cpu_fn(_pt_wa,  warmup, runs)
    ref_lse, pt_lse_ms = _time_cpu_fn(_pt_lse, warmup, runs)

    print(f"\n  [PyTorch CPU reference]")
    print(f"    WA  = {ref_wa:16.4f}   ({pt_wa_ms:.1f} ms median)")
    print(f"    LSE = {ref_lse:16.4f}   ({pt_lse_ms:.1f} ms median)")

    # ── DREAMPlace CPU reference ─────────────────────────────────────────────
    print(f"\n  [DREAMPlace compiled CPU ops]")
    if HAS_DP_WA:
        dp_wa_val, dp_wa_ms = _time_cpu_fn(lambda: dp_wa_ref(data, gamma), warmup, runs)
        wa_vs_pt = abs(dp_wa_val - ref_wa) / (abs(ref_wa) + 1e-15)
        print(f"    WA  = {dp_wa_val:16.4f}   ({dp_wa_ms:.1f} ms)   "
              f"rel_err_vs_PyTorch: {wa_vs_pt:.2e}")
    else:
        dp_wa_val = None
        print(f"    WA  — not available (DREAMPlace extensions not installed)")

    if HAS_DP_LSE:
        dp_lse_val, dp_lse_ms = _time_cpu_fn(lambda: dp_lse_ref(data, gamma), warmup, runs)
        if dp_lse_val is not None:
            lse_vs_pt = abs(dp_lse_val - ref_lse) / (abs(ref_lse) + 1e-15)
            print(f"    LSE = {dp_lse_val:16.4f}   ({dp_lse_ms:.1f} ms)   "
                  f"rel_err_vs_PyTorch: {lse_vs_pt:.2e}")
        else:
            dp_lse_val = None
    else:
        dp_lse_val = None
        print(f"    LSE — not available (DREAMPlace extensions not installed)")

    # Golden values for error comparison: prefer DREAMPlace (compiled C++) over PyTorch
    gold_wa  = dp_wa_val  if dp_wa_val  is not None else ref_wa
    gold_lse = dp_lse_val if dp_lse_val is not None else ref_lse
    gold_src = "DREAMPlace C++" if dp_wa_val is not None else "PyTorch reference"
    print(f"\n  (Error baseline: {gold_src})")

    # ── TTNN ─────────────────────────────────────────────────────────────────
    if not HAS_TTNN:
        print(f"\n  [TTNN] not available — install ttnn and run with TT hardware.")
        return

    dev = ttnn.open_device(device_id=0)
    nc_p = data["nc_p"]
    ones_wa_tt  = _make_ones_tt(nc_p, 8, dev)
    ones_lse_tt = _make_ones_tt(nc_p, 4, dev)

    def _run_wa():
        return ttnn_wa(
            data["x_nm"], data["y_nm"], data["mask_nm"],
            data["net_weights"], data["net_mask"],
            gamma, data["num_nets"], nc_p, dev, ones_wa_tt,
        )
    def _run_lse():
        return ttnn_lse(
            data["x_nm"], data["y_nm"], data["mask_nm"],
            data["net_weights"], data["net_mask"],
            gamma, data["num_nets"], nc_p, dev, ones_lse_tt,
        )
    def _run_wa_opt():
        return ttnn_wa_opt(
            data["x_nm"], data["y_nm"], data["mask_nm"],
            data["net_weights"], data["net_mask"],
            gamma, data["num_nets"], dev,
        )
    def _run_lse_opt():
        return ttnn_lse_opt(
            data["x_nm"], data["y_nm"], data["mask_nm"],
            data["net_weights"], data["net_mask"],
            gamma, data["num_nets"], dev,
        )

    # Warm-up all four variants
    for _ in range(warmup):
        _run_wa(); _run_lse(); _run_wa_opt(); _run_lse_opt()

    # Timed runs
    wa_vals, wa_t   = [], []
    lse_vals, lse_t = [], []
    wa_o_vals, wa_o_t   = [], []
    lse_o_vals, lse_o_t = [], []
    for _ in range(runs):
        v, t = _run_wa();      wa_vals.append(v);     wa_t.append(t)
        v, t = _run_lse();     lse_vals.append(v);    lse_t.append(t)
        v, t = _run_wa_opt();  wa_o_vals.append(v);   wa_o_t.append(t)
        v, t = _run_lse_opt(); lse_o_vals.append(v);  lse_o_t.append(t)

    ttnn.deallocate(ones_wa_tt);  ttnn.deallocate(ones_lse_tt)
    ttnn.close_device(dev)

    def med(lst): return float(np.median(lst))

    dp_wa_ms_val  = dp_wa_ms  if HAS_DP_WA  else None
    dp_lse_ms_val = dp_lse_ms if HAS_DP_LSE else None

    def report(tag, vals, timings, gold, dp_ms):
        val      = med(vals)
        up_ms    = med([t["upload"]  for t in timings]) * 1000
        comp_ms  = med([t["compute"] for t in timings]) * 1000
        dl_ms    = med([t["download"]for t in timings]) * 1000
        total_ms = up_ms + comp_ms + dl_ms
        abs_err  = abs(val - gold)
        rel_err  = abs_err / (abs(gold) + 1e-15)
        vs_dp    = f"  vs DREAMPlace compute: {comp_ms/dp_ms:.2f}x slower" if dp_ms else ""
        print(f"\n    {tag}  val={val:.1f}  err={rel_err:.2e}  "
              f"({'PASS ✓' if rel_err < 1e-2 else 'FAIL ✗'})")
        print(f"    {tag}  upload={up_ms:.1f}ms  compute={comp_ms:.1f}ms  "
              f"download={dl_ms:.1f}ms  total={total_ms:.1f}ms{vs_dp}")

    print(f"\n  [TTNN baseline  (GEMM reduction)]")
    report("WA ",  wa_vals,  wa_t,  gold_wa,  dp_wa_ms_val)
    report("LSE", lse_vals, lse_t, gold_lse, dp_lse_ms_val)

    print(f"\n  [TTNN optimised  (sum + x/y stacked + no mid-pipeline stalls)]")
    report("WA ",  wa_o_vals,  wa_o_t,  gold_wa,  dp_wa_ms_val)
    report("LSE", lse_o_vals, lse_o_t, gold_lse, dp_lse_ms_val)

    # Summary speedup table
    print(f"\n  {'':20s} {'baseline compute':>18}  {'opt compute':>12}  {'speedup':>8}  {'vs DreamPlace':>14}")
    for model, base_t, opt_t, dp_ms in [
        ("WA",  wa_t,  wa_o_t,  dp_wa_ms_val),
        ("LSE", lse_t, lse_o_t, dp_lse_ms_val),
    ]:
        bc = med([t["compute"] for t in base_t]) * 1000
        oc = med([t["compute"] for t in opt_t])  * 1000
        sp = bc / oc if oc > 0 else float("nan")
        vs = f"{oc/dp_ms:.2f}x slower" if dp_ms else "—"
        print(f"  {model:20s} {bc:>16.1f}ms  {oc:>10.1f}ms  {sp:>7.2f}x  {vs:>14}")


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    # gamma ≈ 10 × base_gamma for a 512-bin, 6000-unit layout:
    #   bin_size ≈ 6000/512 ≈ 11.7 → base_gamma ≈ 1.5 → initial gamma ≈ 15.
    # This is the early-stage value; DREAMPlace anneals it downward over iterations.
    gamma = 15.0

    print("TTNN Smooth Wirelength Benchmark: Weighted-Average + Log-Sum-Exp")
    print(f"  TTNN available      : {HAS_TTNN}")
    print(f"  DREAMPlace WA op    : {HAS_DP_WA}")
    print(f"  DREAMPlace LSE op   : {HAS_DP_LSE}")
    print(f"  gamma               : {gamma}")
    print(f"  BIG sentinel        : {BIG}  (layout_wh=6000 so coords < BIG)")
    print(f"  dtype on device     : float32")
    print()
    print("  NOTE: TTNN computes the forward scalar only.")
    print("  Backward (gradient) stays on CPU via DREAMPlace's compiled autograd.")

    sizes = [
        (  10_000, "small (10K nets)"),
        ( 221_142, "adaptec1 (221K nets)"),
        ( 466_151, "adaptec3 (466K nets)"),
    ]

    for num_nets, label in sizes:
        run_benchmark(num_nets, label, gamma)

    print("\nDone.")


if __name__ == "__main__":
    main()
