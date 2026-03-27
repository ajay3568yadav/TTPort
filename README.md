# TTPort — Self-contained DREAMPlace TTNN Profiling Repository

`TTPort` is a clean, self-contained folder inside `TTPlace` that bundles every
file you need to run, study, and reproduce the TTNN-accelerated DREAMPlace
operator benchmarks.

```
TTPort/
├── README.md                          ← this file
├── benchmarks/                        ← profiling entry-points
│   ├── density_scatter/               ← density map scatter (CPU C++ / PyTorch / TTNN / v6 kernel)
│   ├── wirelength/                    ← WA / LSE wirelength (CPU PyTorch / TTNN)
│   └── field_solver/                  ← Poisson / DCT field solver (CPU DCT / TTNN)
├── tt_kernels/                        ← TT Metal custom kernel source + Python launcher
│   ├── density_scatter_v6.cpp         ← fixed-point, wide-page Metal dataflow kernel
│   └── v6_kernel_launcher.py          ← Python host that packs cell data and launches the kernel
└── dreamplace_ref/                    ← DREAMPlace Python reference ops (package root for PYTHONPATH)
    └── dreamplace/
        ├── configure.py               ← minimal config stub (CUDA_FOUND = FALSE)
        ├── ops/
        │   ├── electric_potential/
        │   │   ├── electric_overflow.py          ← CPU C++ density scatter wrapper (needs compiled ext)
        │   │   ├── ttnn_density_map_scatter.py   ← TTNN density scatter (matmul reformulation)
        │   │   ├── ttnn_poisson_solver.py        ← TTNN field solver (DCT as matmul)
        │   │   └── src/                          ← C++ source for reference / compilation
        │   └── dct/
        │       ├── dct2_fft2.py                  ← DCT2 / IDCT via FFT (needs dct2_fft2_cpp)
        │       ├── discrete_spectral_transform.py← pure-Python DCT building blocks
        │       └── ...
```

---

## Quick-start

### 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python ≥ 3.9 | Tested with 3.10 |
| PyTorch ≥ 2.0 | CPU-only is fine for pure-Python paths |
| `ttnn` package | Required for all TTNN and TT Metal paths |
| TT Metal stack | Required for the custom v6 kernel; `TT_METAL_HOME` must be set |
| DREAMPlace compiled extensions | **Optional** — only needed for CPU C++ reference paths |

### 2. Run `setup.sh` (one-time)

```bash
cd TTPort
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

This makes TTPort **fully self-contained** — no files outside this folder are
needed after this runs. The script:

- Copies the compiled DREAMPlace C++ extensions (`.so` files) into
  `dreamplace_ref/` so they are physically inside TTPort
- Verifies all key imports and prints an `[OK]` / `[!!]` table

**On a fresh machine** (no pre-built DREAMPlace available):

The `build_extensions.py` script (called automatically by `setup.sh`) compiles
the 6 needed extensions from the C++ sources in `dreamplace_ref/.../src/` using
`torch.utils.cpp_extension` — requires only `g++` and PyTorch.

### 3. PYTHONPATH (optional override)

All benchmark scripts auto-add `TTPort/dreamplace_ref` to `sys.path` at
runtime, so no manual `PYTHONPATH` is needed after `setup.sh` has been run.

If you need to override, set it explicitly:

```bash
export PYTHONPATH=/path/to/TTPort/dreamplace_ref:$PYTHONPATH
```

If a compiled extension is not found, the benchmark skips that path and prints
an explanatory message rather than aborting.

---

## Benchmarks

### Density Map Scatter

Compares six paths for computing the density map:

| # | Path | Needs compiled ext? |
|---|------|---------------------|
| 1 | CPU C++ (`ElectricDensityMapFunction`) | Yes (`electric_potential_cpp`) |
| 2 | CPU PyTorch (pure Python triangle formula) | No |
| 3 | TTNN original (on-device overlap + matmul) | No |
| 4 | TTNN accurate (CPU overlap + TT matmul) | No |
| 5 | TTNN `scatter_add` | No |
| 6 | TT Metal custom v6 kernel (fixed-point) | No (kernel compiled on first run) |

```bash
cd TTPort/benchmarks/density_scatter

# Basic run (all paths except v6):
python profile_density_scatter_cpu_vs_ttnn.py

# Include the custom v6 Metal kernel:
python profile_density_scatter_cpu_vs_ttnn.py --run-v6-kernel

# Smaller grid for quick iteration:
python profile_density_scatter_cpu_vs_ttnn.py --num-bins-x 128 --num-bins-y 128 --run-v6-kernel

# Skip slow TTNN paths:
python profile_density_scatter_cpu_vs_ttnn.py --skip-ttnn --skip-ttnn-accur
```

See `benchmarks/density_scatter/README.md` for full CLI reference and v6
kernel troubleshooting.

---

### WA / LSE Wirelength

Benchmarks Weighted-Average (WA) and Log-Sum-Exp (LSE) smooth HPWL against
the DREAMPlace compiled CPU op (optional) and a pure-PyTorch reference.

```bash
cd TTPort/benchmarks/wirelength

# Run standalone benchmark (no external files needed):
python wa_wirelength_benchmark.py

# Or via the profile launcher (identical, also forwards CLI flags):
python profile_wa_wirelength.py
```

The script is fully self-contained; the DREAMPlace compiled op is attempted but
gracefully skipped if not present.

---

### Poisson / DCT Field Solver

Compares the DREAMPlace CPU DCT path (`dct2_fft2`) against `TTNNFieldSolver`.

```bash
cd TTPort/benchmarks/field_solver

# Default 512×512 grid:
python poisson_solver_benchmark.py

# Custom grid size:
python poisson_solver_benchmark.py --num-bins-x 256 --num-bins-y 256

# Skip TTNN (CPU-only sanity):
python poisson_solver_benchmark.py --skip-ttnn

# Include dense matmul vs DREAMPlace DCT sanity check:
python poisson_solver_benchmark.py --compare-matmul-reference
```

> **Note:** The DREAMPlace DCT path requires `dct2_fft2_cpp`. If not built,
> set `PYTHONPATH` to the full DREAMPlace build directory.

---

## TT Metal Custom Kernel (v6)

The custom density scatter kernel is written in C++ for TT Metal's dataflow
engine (`tt_kernels/density_scatter_v6.cpp`). It uses:

- **Fixed-point arithmetic** (scale = 1 × 10⁶) for integer DRAM scatter
- **Wide DRAM pages** for coalesced bandwidth
- **Multi-core partitioning** — one Tensix core per cell partition

The Python host launcher (`tt_kernels/v6_kernel_launcher.py`) packs cell data
into buffers, sets kernel compile-time arguments, and dispatches the program.

### Running the v6 kernel directly

```bash
# The density scatter benchmark with --run-v6-kernel handles everything:
cd TTPort/benchmarks/density_scatter
python profile_density_scatter_cpu_vs_ttnn.py --run-v6-kernel
```

`TT_METAL_HOME` is auto-set by `v6_kernel_launcher.py` to the `tt_kernels/`
directory when the kernel file is found there. You only need to set it manually
if the auto-detection fails:

```bash
export TT_METAL_HOME=/path/to/TTPort/tt_kernels
```

---

## TTNN Implementations (dreamplace_ref)

| File | What it does |
|---|---|
| `dreamplace/ops/electric_potential/ttnn_density_map_scatter.py` | Density map scatter via TTNN matmul reformulation or `scatter_add` |
| `dreamplace/ops/electric_potential/ttnn_poisson_solver.py` | Spectral Poisson solver (DCT-as-matmul) on Tenstorrent hardware |

These files have no dependencies beyond `torch` and `ttnn`. Import them
directly or via `PYTHONPATH=TTPort/dreamplace_ref`.

---

## DREAMPlace Reference Notes

After `bash setup.sh` the `dreamplace_ref/` subtree is **fully self-contained**:

| What | Where inside TTPort |
|---|---|
| Python wrappers | `dreamplace_ref/dreamplace/ops/*/` |
| Compiled `.so` extensions | same directories (copied by `setup.sh`) |
| C++ source files | `dreamplace_ref/dreamplace/ops/*/src/` |
| Build script | `build_extensions.py` (called by `setup.sh --source-only`) |

No external DREAMPlace tree is required after setup. If you clone TTPort on a
new machine:

```bash
# Requires: g++ and pip install torch
bash setup.sh          # auto-detects pre-built .so or builds from src
```
