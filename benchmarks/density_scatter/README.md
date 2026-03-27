# Density Map Scatter Benchmark (TTPlace)

This document explains how to run the density map scatter benchmark for all implementations currently used in TTPlace, including the custom v6 TT kernel path.

Benchmark script:

- `TTPlace/dreamplace_ttnn_profile/scripts/profile_density_map_scatter_cpu_vs_ttnn.py`

## What this benchmark includes

The script compares these paths:

- CPU C++ reference (`ElectricDensityMapFunction` from DREAMPlace)
- CPU PyTorch reference (exact triangle overlap formula)
- TTNN original (`density_map_scatter_ttnn`, on-device overlap + matmul)
- TTNN accurate (`density_map_scatter_ttnn_accurate`, CPU overlap + TT matmul)
- TTNN `scatter_add` path (native `ttnn.scatter_add`, may fail for large shapes due to L1 limits)
- TT custom v6 kernel (`--run-v6-kernel`)

## Prerequisites

Run from your TTPlace virtual environment where `torch` and `ttnn` are installed.

Set `PYTHONPATH` so DREAMPlace modules resolve:

```bash
export PYTHONPATH=/home/ubuntu/ayadav/TT_Port/TTPlace/DREAMPlace:$PYTHONPATH
```

`TT_METAL_HOME` is optional for this benchmark script, because the v6 launcher now uses an absolute kernel path internally. You can still set it if desired:

```bash
export TT_METAL_HOME=/home/ubuntu/ayadav/TT_Port/TTPlace/DREAMPlace
```

## Run the full benchmark (all available paths)

```bash
cd /home/ubuntu/ayadav/TT_Port/TTPlace/dreamplace_ttnn_profile/scripts
python profile_density_map_scatter_cpu_vs_ttnn.py --num-bins-x 512 --num-bins-y 512 --run-v6-kernel
```

This runs CPU, TTNN original, TTNN accurate, TTNN scatter_add (if supported for shape), and v6.

## Run only the v6 kernel path

Use skip flags to disable other TT paths:

```bash
cd /home/ubuntu/ayadav/TT_Port/TTPlace/dreamplace_ttnn_profile/scripts
python profile_density_map_scatter_cpu_vs_ttnn.py \
  --num-bins-x 512 --num-bins-y 512 \
  --skip-ttnn-orig --skip-ttnn-accur --skip-ttnn-scatter-add \
  --run-v6-kernel
```

## Quick sanity run (faster debug shape)

```bash
cd /home/ubuntu/ayadav/TT_Port/TTPlace/dreamplace_ttnn_profile/scripts
python profile_density_map_scatter_cpu_vs_ttnn.py \
  --num-bins-x 128 --num-bins-y 128 \
  --num-movable 20000 --num-fixed 1000 --num-filler 2000 \
  --warmup 1 --iters 2 \
  --skip-ttnn-orig --skip-ttnn-accur --skip-ttnn-scatter-add \
  --run-v6-kernel
```

## Useful flags

- `--warmup <int>`: warmup iterations before timing
- `--iters <int>`: timed iterations
- `--chunk-size <int>`: chunk size for TT matmul paths
- `--skip-ttnn-orig`
- `--skip-ttnn-accur`
- `--skip-ttnn-scatter-add`
- `--run-v6-kernel`

## Expected behavior / interpretation notes

- TTNN `scatter_add` may report unsupported for large shapes (L1 circular buffer limit). This is expected and handled gracefully by the script.
- v6 kernel now loads via absolute path from:
  - `TTPlace/DREAMPlace/tt_metal/kernels/dataflow/density_scatter_v6.cpp`
- If v6 launches correctly, you should see:
  - `[6/6] TT custom v6 kernel (fixed-point + wide pages) ...`
  - timing stats
  - accuracy stats vs CPU C++ (`rel_l2`, `max_abs`, etc.)

## Troubleshooting (v6-specific)

If v6 still does not run:

1. Confirm file exists:
   ```bash
   ls /home/ubuntu/ayadav/TT_Port/TTPlace/DREAMPlace/tt_metal/kernels/dataflow/density_scatter_v6.cpp
   ```
2. Confirm benchmark can import helper:
   - `TTPlace/DREAMPlace/tt_custom_kernel_scatter_benchmark.py`
3. Re-run with only v6 enabled (command above) to isolate errors.
4. If error is from TT runtime resource limits, reduce shape (for example `128x128`) and verify correctness first.

