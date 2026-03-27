# SPDX-FileCopyrightText: © 2026
# SPDX-License-Identifier: Apache-2.0
#
# Tenstorrent WA wirelength *forward* for training loss, with DREAMPlace merged
# *backward* (CPU analytical kernel using grad_intermediate from CPU forward).
#
# Forward scalar WL is computed on TT via ``ttnn_wa_opt`` (TTNN ops) only — no custom tt-metal kernels.
# Each step runs CPU merged forward once to obtain grad_intermediate for backward.
# If TT forward fails, falls back to CPU WL for that step.

from __future__ import annotations

import logging
import time
from typing import Dict, Type

import torch
from torch.autograd import Function

import dreamplace.ops.weighted_average_wirelength.weighted_average_wirelength_cpp_merged as weighted_average_wirelength_cpp_merged
from dreamplace.ops.wirelength_tt.tt_metric import _import_ttnn_wa_opt, compute_tt_wl_scalar_pins

logger = logging.getLogger(__name__)

_ttnn_wa_opt_cache = None


def _get_ttnn_wa_opt():
    global _ttnn_wa_opt_cache
    if _ttnn_wa_opt_cache is None:
        _ttnn_wa_opt_cache = _import_ttnn_wa_opt()
    return _ttnn_wa_opt_cache


def make_weighted_average_wirelength_merged_tt_class(tt_device):
    """WA forward on TT via ``ttnn_wa_opt`` (TTNN) only."""

    ttnn_wa_opt_fn = _get_ttnn_wa_opt()

    class WeightedAverageWirelengthMergedTTFunction(Function):
        @staticmethod
        def forward(
            ctx,
            pos,
            flat_netpin,
            netpin_start,
            pin2net_map,
            net_weights,
            net_mask,
            pin_mask,
            inv_gamma,
        ):
            if pos.is_cuda:
                raise RuntimeError("use_tt_merged_wa_forward requires CPU pin positions")

            func = weighted_average_wirelength_cpp_merged.forward
            t0 = time.perf_counter()
            output = func(
                pos.view(pos.numel()),
                flat_netpin,
                netpin_start,
                pin2net_map,
                net_weights,
                net_mask,
                inv_gamma,
            )
            t_cpu = (time.perf_counter() - t0) * 1000
            wl_cpu = output[0]
            ctx.grad_intermediate = output[1]
            ctx.pos = pos
            ctx.flat_netpin = flat_netpin
            ctx.netpin_start = netpin_start
            ctx.pin2net_map = pin2net_map
            ctx.net_weights = net_weights
            ctx.net_mask = net_mask
            ctx.pin_mask = pin_mask
            ctx.inv_gamma = inv_gamma

            num_nets = int(netpin_start.numel() - 1)
            gamma = float(1.0 / inv_gamma.reshape(-1)[0].item())

            wl_tt = None
            try:
                t1 = time.perf_counter()
                wl_tt = compute_tt_wl_scalar_pins(
                    pos,
                    flat_netpin,
                    netpin_start,
                    num_nets,
                    net_weights,
                    net_mask,
                    gamma,
                    tt_device,
                    ttnn_wa_opt_fn,
                )
                logger.debug(
                    "TT WL forward %.3f ms (CPU merged forward %.3f ms)",
                    (time.perf_counter() - t1) * 1000,
                    t_cpu,
                )
            except Exception as e:
                logger.warning("TT WA forward failed, using CPU WL for this step: %s", e)
                return wl_cpu

            return pos.new_tensor(wl_tt, dtype=pos.dtype, device=pos.device)

        @staticmethod
        def backward(ctx, grad_wl):
            func = weighted_average_wirelength_cpp_merged.backward
            grad_out = func(
                grad_wl,
                ctx.pos,
                ctx.grad_intermediate,
                ctx.flat_netpin,
                ctx.netpin_start,
                ctx.pin2net_map,
                ctx.net_weights,
                ctx.net_mask,
                ctx.inv_gamma,
            )
            pin_mask_bool = (
                ctx.pin_mask.to(torch.bool) if ctx.pin_mask.dtype != torch.bool else ctx.pin_mask
            )
            n2 = int(grad_out.numel() // 2)
            grad_out[:n2].masked_fill_(pin_mask_bool, 0.0)
            grad_out[n2:].masked_fill_(pin_mask_bool, 0.0)
            return grad_out, None, None, None, None, None, None, None

    return WeightedAverageWirelengthMergedTTFunction


_merged_tt_class_cache: Dict[int, Type] = {}


def get_merged_tt_function_class(tt_device):
    """Cache one autograd Function class per device."""
    key = id(tt_device)
    if key not in _merged_tt_class_cache:
        _merged_tt_class_cache[key] = make_weighted_average_wirelength_merged_tt_class(tt_device)
    return _merged_tt_class_cache[key]
