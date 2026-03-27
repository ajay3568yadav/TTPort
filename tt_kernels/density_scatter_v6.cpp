// SPDX-FileCopyrightText: © 2025 DREAMPlace / TT scatter port
// SPDX-License-Identifier: Apache-2.0
//
// Density map scatter — v6 (fixed-point cell coords + float ratio), BRISC dataflow.
// Matches tt_custom_kernel_scatter_benchmark.run_tt_kernel_v6() host launch.
//
// Runtime args (per core):
//   0 cell_base_addr   1 out_base_addr   2 n_cells   3 num_bins_x   4 num_bins_y
//   5 row_stride_bytes (= num_bins_y * 4)   6 xl_fp   7 yl_fp   8 bsx_fp   9 bsy_fp
//   10 page_start   11 page_end   12 out_row_offset
// SCALE=1024 for fixed-point, same as Python V6_SCALE.

#include <cstdint>

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

namespace {

constexpr uint32_t kScale = 1024;
constexpr uint32_t kCellsPerPage = 64;
constexpr uint32_t kFloatsPerCell = 8;
constexpr uint32_t cb_grid = 0;
constexpr uint32_t cb_page = 1;

FORCE_INLINE float fmax3(float a, float b) { return (a > b) ? a : b; }
FORCE_INLINE float fmin3(float a, float b) { return (a < b) ? a : b; }

FORCE_INLINE int32_t fp_bits_to_int32(float f) {
    union {
        float f;
        int32_t i;
    } x;
    x.f = f;
    return x.i;
}

}  // namespace

void kernel_main() {
    const uint32_t cell_base = get_arg_val<uint32_t>(0);
    const uint32_t out_base = get_arg_val<uint32_t>(1);
    const uint32_t n_cells = get_arg_val<uint32_t>(2);
    const uint32_t num_bins_x = get_arg_val<uint32_t>(3);
    const uint32_t num_bins_y = get_arg_val<uint32_t>(4);
    const uint32_t row_stride = get_arg_val<uint32_t>(5);
    const int32_t xl_fp = static_cast<int32_t>(get_arg_val<uint32_t>(6));
    const int32_t yl_fp = static_cast<int32_t>(get_arg_val<uint32_t>(7));
    const int32_t bsx_fp = static_cast<int32_t>(get_arg_val<uint32_t>(8));
    const int32_t bsy_fp = static_cast<int32_t>(get_arg_val<uint32_t>(9));
    const uint32_t page_start = get_arg_val<uint32_t>(10);
    const uint32_t page_end = get_arg_val<uint32_t>(11);
    const uint32_t out_row_offset = get_arg_val<uint32_t>(12);

    constexpr auto cell_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<TensorAccessorArgs<0>::next_compile_time_args_offset()>();

    const uint32_t cell_page_bytes = kCellsPerPage * kFloatsPerCell * sizeof(float);
    const uint32_t grid_elems = num_bins_x * num_bins_y;

    const TensorAccessor s_cell(cell_args, cell_base, cell_page_bytes);
    const TensorAccessor s_out(out_args, out_base, row_stride);

    const uint8_t noc = noc_index;

    cb_reserve_back(cb_grid, 1);
    {
        uint32_t wptr = get_write_ptr(cb_grid);
        volatile float* gw = reinterpret_cast<volatile float*>(wptr);
        for (uint32_t i = 0; i < grid_elems; i++) gw[i] = 0.0f;
    }
    cb_push_back(cb_grid, 1);
    cb_wait_front(cb_grid, 1);
    float* grid = reinterpret_cast<float*>(get_read_ptr(cb_grid));

    const float inv_scale = 1.0f / static_cast<float>(kScale);
    const float xl = static_cast<float>(xl_fp) * inv_scale;
    const float yl = static_cast<float>(yl_fp) * inv_scale;
    const float bsx = static_cast<float>(bsx_fp) * inv_scale;
    const float bsy = static_cast<float>(bsy_fp) * inv_scale;
    const float inv_bsx = 1.0f / bsx;
    const float inv_bsy = 1.0f / bsy;

    for (uint32_t p = page_start; p < page_end; p++) {
        cb_reserve_back(cb_page, 1);
        uint32_t l1_page = get_write_ptr(cb_page);
        uint64_t src_noc = s_cell.get_noc_addr(p, 0, noc);
        noc_async_read(src_noc, l1_page, cell_page_bytes, noc);
        noc_async_read_barrier();
        cb_push_back(cb_page, 1);
        cb_wait_front(cb_page, 1);

        float* page = reinterpret_cast<float*>(get_read_ptr(cb_page));
        for (uint32_t c = 0; c < kCellsPerPage; c++) {
            const uint32_t gidx = p * kCellsPerPage + c;
            if (gidx >= n_cells) break;

            float* cell = page + c * kFloatsPerCell;
            const int32_t nx_i = fp_bits_to_int32(cell[0]);
            const int32_t ny_i = fp_bits_to_int32(cell[1]);
            const int32_t szx_i = fp_bits_to_int32(cell[2]);
            const int32_t szy_i = fp_bits_to_int32(cell[3]);
            const float ratio = cell[6];

            const float nx = static_cast<float>(nx_i) * inv_scale;
            const float ny = static_cast<float>(ny_i) * inv_scale;
            const float szx = static_cast<float>(szx_i) * inv_scale;
            const float szy = static_cast<float>(szy_i) * inv_scale;

            int32_t bxl = static_cast<int32_t>((nx - xl) * inv_bsx);
            if (bxl < 0) bxl = 0;
            uint32_t ubxl = static_cast<uint32_t>(bxl);
            uint32_t bxh = static_cast<uint32_t>(((nx + szx - xl) * inv_bsx)) + 1;
            if (bxh > num_bins_x) bxh = num_bins_x;

            int32_t byl = static_cast<int32_t>((ny - yl) * inv_bsy);
            if (byl < 0) byl = 0;
            uint32_t ubyl = static_cast<uint32_t>(byl);
            uint32_t byh = static_cast<uint32_t>(((ny + szy - yl) * inv_bsy)) + 1;
            if (byh > num_bins_y) byh = num_bins_y;

            for (uint32_t bx = ubxl; bx < bxh; bx++) {
                const float bin_xl = xl + static_cast<float>(bx) * bsx;
                const float px = fmax3(0.0f, fmin3(nx + szx, bin_xl + bsx) - fmax3(nx, bin_xl));
                const float pxr = px * ratio;
                for (uint32_t by = ubyl; by < byh; by++) {
                    const float bin_yl = yl + static_cast<float>(by) * bsy;
                    const float py = fmax3(0.0f, fmin3(ny + szy, bin_yl + bsy) - fmax3(ny, bin_yl));
                    grid[bx * num_bins_y + by] += pxr * py;
                }
            }
        }
        cb_pop_front(cb_page, 1);
    }

    for (uint32_t r = 0; r < num_bins_x; r++) {
        uint32_t l1_row = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(grid + r * num_bins_y));
        uint64_t dst = s_out.get_noc_addr(out_row_offset + r, 0, noc);
        noc_async_write(l1_row, dst, row_stride, noc);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_grid, 1);
}
