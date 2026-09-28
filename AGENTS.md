# Repository Guidelines

## Local Agent Instructions

Before working in this repository, read `AGENTS.local.md` if it exists. It
contains machine-specific paths, hardware, and environment instructions that
are intentionally excluded from version control. Apply those local
instructions together with this file; if the file is absent, continue using
this document alone.

## Project Structure & Module Organization

- `include/gunrock/` contains the header-only graph library, formats, I/O, backends, and algorithms.
- `examples/algorithms/` contains runnable GPU examples; `benchmarks/` contains NVBench drivers.
- `unittests/` contains GoogleTest tests, `python/` contains PyGunrock, and `datasets/` contains Matrix Market recipes and fixtures.
- `cmake/`, `docs/`, and `scripts/` contain dependency helpers, documentation, and tooling.

## Environment Setup

Native CUDA benchmarks do not require or use a Python virtual environment. Run CMake and benchmark commands from the repository root with the system CUDA Toolkit, compiler, CMake, and NVIDIA driver. Do not activate a venv for this workflow. Verify prerequisites with `nvcc --version` and `cmake --version`; a working GPU driver is required at runtime. CMake fetches the pinned NVBench revision automatically.

The optional `python/` package is separate and requires Python >=3.9. For it, use `python3.10 -m venv .venv`, `source .venv/bin/activate`, and `python -m pip install -r python/requirements.txt`. This environment is not needed for `benchmarks/*.cu`.

## Build, Test, and Development Commands

Configure an NVIDIA build with your GPU architecture (`90` is H100; see `README.md` for common codes):

```bash
cmake -S . -B build \
    -DESSENTIALS_NVIDIA_BACKEND=ON -DESSENTIALS_AMD_BACKEND=OFF \
    -DESSENTIALS_BUILD_BENCHMARKS=ON -DESSENTIALS_BUILD_EXAMPLES=OFF \
    -DESSENTIALS_BUILD_TESTS=OFF -DNVBench_ENABLE_CUPTI=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target tc_bench -j"$(nproc)"
```

Replace `86` with your GPU architecture. Replace `tc_bench` to build another target, such as `bfs_bench`; executables are written to `build/bin/`. Run Triangle Counting on one GPU with `./build/bin/tc_bench --devices 0 --market datasets/roadNet-CA/roadNet-CA.mtx --reduce true`. Download standard matrices with `make -C datasets STANDARD`. Run native tests with `ctest --test-dir build --output-on-failure`. Run Python tests from `python/` with `pytest tests/ -v` inside `.venv`.

Generate full benchmark set by `for BENCHNAME in {bc,bfs,color,geo,hits,kcore,mst,ppr,pr,spgemm,spmv,sssp,tc}_bench; do cmake --build build --target $BENCHNAME -j$(nproc); done`.

## Coding Style & Naming Conventions

Use C++17, two-space indentation, snake_case identifiers, and existing suffixes such as `.cu`, `.cuh`, and `.hxx`. Use `scripts/format.sh . y`; preserve include ordering because automatic sorting is disabled.

## Testing Guidelines

Add focused GoogleTest cases under the matching `unittests/` directory, using names such as `TEST(algorithm, tc)`. Add Python tests under `python/tests/` with `test_*.py`; exercise both backends when relevant.

## Commit & Pull Request Guidelines

Use short, imperative commit subjects such as `Add ...` or `Enable ...`. Pull requests should describe backend/GPU impact, validation commands and results, and dataset requirements. Do not commit build outputs, downloaded datasets, or external dependency trees.
