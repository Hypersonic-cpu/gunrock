# Merge-path software prefetch

BFS, SSSP, and both BC traversals support NVIDIA cache prefetch instructions in
`merge_path` advance. Select `--advance=merge_path` for BFS/SSSP; BC already uses
this traversal. Existing callers and CLI defaults use `--swpf=none`.

```bash
./build-v2/bin/bfs --binary-in graph.bin --src 13331 \
  --advance=merge_path --swpf=spp --swpf-target=l2 --swpf-distance=2
```

Supported algorithms are `none`, `gp`, and `spp`. Targets are `l1` and `l2`;
distances are `1`, `2`, `4`, and `8` (default `2`). The target defaults to `l2`.

The merge scheduler supplies actual edge records, skipping row boundaries and
invalid frontier entries. GP gathers up to `distance` edges, prefetches CSR
addresses, resolves destinations and weights, prefetches application state, and
consumes the group in order. SPP uses a small ring: CSR prefetch at step `i`,
destination resolution/state prefetch at `i + distance`, and the original
operation at `i + 2 * distance`. Pipeline drain handles partial groups and empty
rows. Each thread owns at most 11 merge items; large distances may have little
opportunity to overlap work on sparse rows.

Application hooks declare whether an operator uses edge weights. BFS and both
BC phases skip weight prefetches and weight loads in GP/SPP; SSSP retains them
because relaxation uses the weight. Original-loop hint and lookahead controls
apply the same rule. Generic operators without an application hook retain
weighted behavior.

`profiling/validate_swpf_weight_usage.cu` is a compile-only regression for this
policy. Its mock graph deletes both weight accessors; BFS and both BC hooks
must compile for L1/L2 targets, while static assertions require SSSP and generic
hooks to retain weighted behavior. Compile it with the benchmark CUDA flags.

`swpf/` contains the reusable PTX wrappers, policies, and application address
hooks. Read prefetches follow the cache target; atomic targets always use L2.
The actual loads and atomic operations remain in the original application
operators. SSSP source distances are not memoized: a source distance can improve
while the kernel runs. Source distance addresses also prefetch to L2 because
other edges can update them atomically. The `none` specialization retains the original serial
merge loop and omits explicit prefetch instructions and state-hook arguments.

BC validation uses a parallel, level-synchronous CPU Brandes reference with
OpenMP. It matches single-source BC, excluding the source and using the existing
0.5 scale. CPU double intermediates are compared to GPU float output using
`1e-4 + 1e-4 * max(abs(a), abs(b))`. SSSP validation uses relative tolerance
`1e-5`; BFS compares distances exactly. Validation failures exit with nonzero
status. BC benchmark repeats clear the output before each timed run, preserving
the library's API for accumulating contributions across sources. BC frontiers
grow when traversal depth exceeds the initial allocation.

Reproducible validation and measurement:

```bash
cmake --build build-v2 --target bfs sssp bc --parallel 6
python3 profiling/validate_swpf.py --suite-config profiling/swpf_suite.yaml \
  --build-bin build-v2/bin --output-dir RESULTS/validation-small
python3 profiling/validate_swpf.py --suite-config profiling/swpf_suite.yaml \
  --build-bin build-v2/bin --output-dir RESULTS/validation-large --large
python3 profiling/run_native_suite.py --suite-config profiling/swpf_suite.yaml \
  --build-bin build-v2/bin --output-dir RESULTS/native
python3 profiling/swpf_resources.py --build-bin build-v2/bin \
  --output-dir RESULTS/resources
```

The YAML specifies the binary root and filenames, graph dimensions, source,
repeat count, and all prefetch variants. The native runner keeps every one of
10 program-reported CUDA-event algorithm timings and reports median, mean,
standard deviation, and range. Algorithm timing includes all traversal kernels,
scans/filters, and host scheduling inside `enact`; it is not an isolated main
kernel measurement. The first sample is retained without an unreported warmup.

Resource inspection uses `cuobjdump` and CUDA driver attribute/occupancy queries.
It verifies that `none` contains no explicit prefetch, collects compiled register
and local/shared-memory usage, and records theoretical occupancy. It does not
execute a profiler or collect hardware counters. Cache hit rates, DRAM traffic,
long-scoreboard stalls, eligible warps, and achieved occupancy cannot be inferred
from native timings and are not reported as measured results.
