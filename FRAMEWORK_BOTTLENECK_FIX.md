# Gunrock Main BFS Native Profiling + `exp-opt` Optimization Experiment Plan

Machine-specific checkout, dataset, result, GPU, CUDA, and profiler values are
kept in `AGENTS.local.md`. Set the variables documented there before running
the commands in this plan; the commands below use those variables instead of
embedding one developer's absolute paths.

## Goal

Profile the already-built Gunrock `main` branch BFS implementation natively, identify whether repeated `cudaMalloc` / `cudaFree` operations appear between BFS iterations and create large execution gaps, then implement and evaluate an optimized version on a new local branch named:

```bash
exp-opt
```

The experiment should answer the following questions:

1. Does the current BFS implementation repeatedly invoke `cudaMalloc` / `cudaFree` during BFS traversal?
2. Are those allocations caused by the current `block_mapped` advance implementation constructing:

   ```cpp
   thrust::device_vector<typename frontier_t::offset_t> block_offsets(1);
   ```

   on every advance call?
3. Do those runtime allocation/free operations create visible gaps between BFS levels?
4. How much performance improves if only the per-level temporary device allocation is removed?
5. How much additional performance improves if the redundant degree/output-size pre-pass is also removed?
6. After removing software-engineering overhead, how much of the original producer-to-consumer gap remains because of the actual GPU load-balancing algorithm?

We will compare three implementations:

```text
A:
    Stock current-main implementation

B0:
    Reuse a persistent output counter
    Remove per-level temporary device allocation/free
    Keep the current output-size transform_reduce pre-pass

B:
    B0
    +
    Remove the redundant output-size transform_reduce pre-pass
    Determine the actual raw output size from the persistent counter
```

The final research question is:

> After removing Gunrock framework overhead, does the real `row_offsets / degree -> load balancing -> column_indices[e]` producer-consumer separation remain substantial?

---

# 0. Important execution rules

Do not modify the source code before collecting the stock baseline.

The required order is:

```text
Check environment
    ↓
Check repository/build state
    ↓
Locate existing bfs_bench
    ↓
Validate stock BFS
    ↓
Profile stock BFS with Nsight Systems
    ↓
Analyze cudaMalloc/cudaFree behavior
    ↓
Save stock binary/results
    ↓
git checkout -b exp-opt
    ↓
Implement B0
    ↓
Build + validate + profile B0
    ↓
Implement B
    ↓
Build + validate + profile B
    ↓
Native A/B0/B evaluation
    ↓
Run final larger evaluation on soc-LiveJournal1
    ↓
Analyze results
```

Do not change:

- BFS algorithm semantics;
- the current BFS `atomicMin` distance update;
- `block_mapped` as the main load-balancing policy;
- the CTA-local block scan;
- edge-to-source binary search;
- CSR adjacency traversal;
- filter settings during the main A/B comparison.

The main A/B comparison must use:

```text
advance_load_balance = block_mapped
filter disabled
```

Most importantly, **always pass Gunrock's `--profile` flag** when running `bfs_bench`.

The purpose is to disable/avoid Gunrock's built-in performance/profiling framework interfering with the native profiling experiment.

Canonical execution form:

```bash
CUDA_VISIBLE_DEVICES="$TEST_GPU" \
  "$REPO_ROOT/build/bin/bfs_bench" \
  --market "$DATA_ROOT/roadNet-CA.mtx" \
  --profile
```

All baseline, validation, Nsight Systems, Nsight Compute, and native-performance runs should include:

```bash
--profile
```

unless a specific command proves that `--profile` cannot be combined with that operation.

---

# 1. Environment check

Record the machine-specific environment from `AGENTS.local.md` and verify it
again before starting. The current local values are intentionally not copied
into this portable experiment plan.

```text
CUDA_ROOT: value from AGENTS.local.md
Nsight Compute: host-installed version
Nsight Systems: host-installed version
GPU: host hardware and TEST_GPU from AGENTS.local.md
```

Important profiler privilege requirement for this host is recorded in
`AGENTS.local.md`:

```text
Nsight Systems and Nsight Compute must be invoked through:

sudo $(which nsys)
sudo $(which ncu)

The local instructions record whether these commands have password-free sudo
permission; stop if that local prerequisite is not satisfied.
```

Re-check the actual environment before starting:

```bash
pwd

git branch --show-current
git status --short
git rev-parse HEAD
git log -1 --oneline

which nvcc
nvcc --version

echo "$CUDA_HOME"
readlink -f "$CUDA_HOME"

which nsys
sudo "$(which nsys)" --version

which ncu
sudo "$(which ncu)" --version

gcc --version

nvidia-smi

nvidia-smi \
  --query-gpu=index,name,compute_cap,memory.used,memory.total,utilization.gpu \
  --format=csv

nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv
```

Choose the less busy GPU identified by `AGENTS.local.md` and use the same GPU
for every A/B comparison.

For example:

```bash
export TEST_GPU="$GPU_VISIBLE"
```

Ordinary run:

```bash
CUDA_VISIBLE_DEVICES=$TEST_GPU ...
```

Profiler run:

```bash
sudo env CUDA_VISIBLE_DEVICES=$TEST_GPU "$(which nsys)" ...
```

Do not run A and B on different GPUs.

---

# 2. Repository and build-state check

The current working branch is expected to be approximately:

```text
exp-env
```

Check:

```bash
git branch --show-current
git status --porcelain
git rev-parse HEAD
git log -1 --oneline
```

Do not reset or discard any existing user modifications.

If the working tree is dirty, first save a record:

```bash
git diff > /tmp/gunrock-exp-env-before.patch
git status --short > /tmp/gunrock-exp-env-before.status
```

Do not automatically apply, stash, reset, or delete anything unless absolutely necessary.

Verify that the expected current-main implementation is present:

```bash
grep -n "block_offsets" \
  include/gunrock/framework/operators/advance/block_mapped.hxx

grep -n "compute_output_length" \
  include/gunrock/framework/operators/advance/block_mapped.hxx

grep -n "context.synchronize" \
  include/gunrock/framework/operators/advance/block_mapped.hxx

grep -n "transform_reduce" \
  include/gunrock/framework/operators/advance/helpers.hxx

grep -n "scanned_work_domain" \
  include/gunrock/framework/enactor.hxx
```

---

# 3. Source-code behavior that must be understood before modification

## 3.1 BFS execution path

Relevant file:

```text
include/gunrock/algorithms/bfs.hxx
```

The BFS loop performs:

```cpp
operators::advance::execute_runtime(
    G,
    E,
    search,
    advance_load_balance,
    context);
```

The BFS search lambda contains logic equivalent to:

```cpp
auto old_distance =
    math::atomic::min(
        &distances[neighbor],
        iteration + 1);

return iteration + 1 < old_distance;
```

Do not modify this logic.

The main experiment should use:

```text
block_mapped
filter disabled
```

---

## 3.2 Current block-mapped path performs a degree/output-size pre-pass

Relevant file:

```text
include/gunrock/framework/operators/advance/block_mapped.hxx
```

Before launching the real advance kernel, the current implementation calls:

```cpp
auto size_of_output =
    compute_output_length(G, input, context);
```

Relevant implementation:

```text
include/gunrock/framework/operators/advance/helpers.hxx
```

`compute_output_length()` uses a Thrust reduction approximately equivalent to:

```cpp
thrust::transform_reduce(
    frontier,
    degree(vertex),
    plus);
```

Therefore each BFS level currently looks approximately like:

```text
input frontier
    ↓
read vertex degrees / CSR row offsets
    ↓
transform_reduce
    ↓
obtain output size
    ↓
launch real block_mapped advance
```

The real block-mapped kernel then reads the metadata again:

```cpp
sedges[local_idx] =
    G.get_starting_edge(v);

th_deg[0] =
    G.get_number_of_neighbors(v);
```

Thus the CSR row metadata is touched once for sizing and then again for the actual traversal.

---

## 3.3 Current block-mapped path creates a one-element device vector per call

The current code also contains:

```cpp
thrust::device_vector<typename frontier_t::offset_t>
    block_offsets(1);
```

The kernel uses this scalar as a global output counter.

After the launch, the function performs:

```cpp
context.synchronize();
```

The expected stock sequence is therefore roughly:

```text
compute_output_length()
    ↓
Thrust transform_reduce
    ↓
temporary device allocation
    ↓
block_mapped_kernel
    ↓
stream synchronization
    ↓
temporary device deallocation
```

The first profiling stage must verify whether this expectation is actually visible as:

```text
cudaMalloc
cudaFree
```

or related allocation APIs.

Do not assume the result in advance.

---

## 3.4 Persistent GPU scratch storage already exists

Relevant file:

```text
include/gunrock/framework/enactor.hxx
```

The enactor already owns:

```cpp
thrust::device_vector<edge_t>
    scanned_work_domain;
```

initialized approximately as:

```cpp
scanned_work_domain(
    problem->get_graph().get_number_of_vertices() + 1)
```

This allocation occurs at enactor construction time, not on every BFS level.

The default `block_mapped` path does not need a global scanned work domain.

Therefore the optimization experiment can reuse:

```text
scanned_work_domain[0]
```

as a persistent block output counter.

This avoids creating a new device allocation every BFS level.

---

## 3.5 Frontier buffers are already pre-reserved

The enactor constructor currently reserves approximately:

```cpp
max(number_of_edges, number_of_vertices)
```

elements for each frontier buffer.

Therefore a BFS-specific experimental fast path can potentially skip the output-size pre-pass and write directly into the already-reserved output frontier.

This is safe for this BFS experiment only if:

```text
output capacity >= number of graph edges
```

because one BFS level cannot generate more raw adjacency work than the graph contains.

Do not generalize this assumption to every Gunrock operator or algorithm.

If the capacity condition is not satisfied, fall back to the original implementation.

---

# 4. BFS executable

The expected already-built executable is:

```bash
"$REPO_ROOT/build/bin/bfs_bench"
```

Set:

```bash
export BFS_BIN="$REPO_ROOT/build/bin/bfs_bench"
```

Verify:

```bash
test -x "$BFS_BIN"
file "$BFS_BIN"
"$BFS_BIN" --help
```

Confirm that the executable supports at least:

```text
--market
--profile
--advance_load_balance
--validate
```

Use this executable rather than rebuilding the entire project unnecessarily.

---

# 5. Dataset strategy

Use these datasets:

```bash
"$DATA_ROOT/roadNet-CA.mtx"

"$DATA_ROOT/webbase-1M.mtx"

"$DATA_ROOT/soc-LiveJournal1.mtx"
```

Verify them:

```bash
ls -lh \
  "$DATA_ROOT/roadNet-CA.mtx" \
  "$DATA_ROOT/webbase-1M.mtx" \
  "$DATA_ROOT/soc-LiveJournal1.mtx"
```

Use this order:

```text
Phase 1:
    roadNet-CA

Phase 2:
    webbase-1M

Phase 3:
    soc-LiveJournal1
```

Do **not** begin development/debugging with `soc-LiveJournal1`.

Its native runtime is relatively long.

Use:

```text
roadNet-CA
webbase-1M
```

to develop, debug, validate, and profile the optimized implementation.

Only after B is stable and the optimization appears correct should `soc-LiveJournal1` be used for the final evaluation.

Suggested variables:

```bash
export GRAPH_ROAD="$DATA_ROOT/roadNet-CA.mtx"
export GRAPH_WEB="$DATA_ROOT/webbase-1M.mtx"
export GRAPH_LJ="$DATA_ROOT/soc-LiveJournal1.mtx"
```

Unless a source is explicitly required, use Gunrock's default source behavior consistently between A/B0/B.

If a fixed source is needed for deterministic comparison, use the same source for all versions and all repetitions.

---

# 6. Result directories

Create:

```bash
export OUT="$RESULTS_ROOT/native-bfs-opt"

mkdir -p \
  "$OUT/A-stock" \
  "$OUT/B0-persistent-counter" \
  "$OUT/B-final"
```

Record environment:

```bash
{
  date
  hostname
  git rev-parse HEAD
  git log -1 --oneline
  nvcc --version
  sudo "$(which nsys)" --version
  sudo "$(which ncu)" --version
  gcc --version
  nvidia-smi
} > "$OUT/environment.txt"
```

Do not add `.nsys-rep`, SQLite traces, or other profiling artifacts to Git.

---

# 7. Stock A correctness run

Start with `roadNet-CA`.

Use:

```bash
CUDA_VISIBLE_DEVICES=$TEST_GPU \
"$BFS_BIN" \
  --market "$GRAPH_ROAD" \
  --advance_load_balance block_mapped \
  --profile \
  --validate \
  | tee "$OUT/A-stock/roadNet-CA-validate.txt"
```

Confirm correctness.

If validation reports errors, stop and investigate baseline correctness before changing the code.

Then perform a native run without validation:

```bash
CUDA_VISIBLE_DEVICES=$TEST_GPU \
"$BFS_BIN" \
  --market "$GRAPH_ROAD" \
  --advance_load_balance block_mapped \
  --profile \
  | tee "$OUT/A-stock/roadNet-CA-native-one-run.txt"
```

Repeat the same basic validation on `webbase-1M`:

```bash
CUDA_VISIBLE_DEVICES=$TEST_GPU \
"$BFS_BIN" \
  --market "$GRAPH_WEB" \
  --advance_load_balance block_mapped \
  --profile \
  --validate \
  | tee "$OUT/A-stock/webbase-1M-validate.txt"
```

Do not run `soc-LiveJournal1` yet unless necessary.

---

# 8. Stock A Nsight Systems profiling

## 8.1 Inspect CLI options first

Do not assume Nsight Systems 2024.4 option names.

Run:

```bash
sudo "$(which nsys)" profile --help \
  > "$OUT/nsys-profile-help.txt"

sudo "$(which nsys)" stats --help \
  > "$OUT/nsys-stats-help.txt"
```

Inspect:

```bash
grep -Ei \
  "trace|cuda-memory|sample|overwrite|output|backtrace" \
  "$OUT/nsys-profile-help.txt"
```

If `--cuda-memory-usage=true` exists, enable it.

---

# 9. Profile roadNet-CA first

Use:

```bash
sudo env CUDA_VISIBLE_DEVICES=$TEST_GPU \
"$(which nsys)" profile \
  --trace=cuda,osrt,nvtx \
  --sample=none \
  --force-overwrite=true \
  -o "$OUT/A-stock/roadNet-CA" \
  "$BFS_BIN" \
    --market "$GRAPH_ROAD" \
    --advance_load_balance block_mapped \
    --profile
```

If supported, add:

```bash
--cuda-memory-usage=true
```

If an option is unsupported in this Nsight Systems version, remove only that option and continue.

After profiling:

```bash
sudo chown -R "$(id -u):$(id -g)" "$OUT"
```

if required.

Then repeat for `webbase-1M`:

```bash
sudo env CUDA_VISIBLE_DEVICES=$TEST_GPU \
"$(which nsys)" profile \
  --trace=cuda,osrt,nvtx \
  --sample=none \
  --force-overwrite=true \
  -o "$OUT/A-stock/webbase-1M" \
  "$BFS_BIN" \
    --market "$GRAPH_WEB" \
    --advance_load_balance block_mapped \
    --profile
```

Do not profile `soc-LiveJournal1` until the implementation is stable.

---

# 10. Analyze stock CUDA allocation behavior

The first major goal is to establish whether repeated CUDA allocation/free calls actually occur.

## 10.1 CUDA API summary

Inspect available reports:

```bash
sudo "$(which nsys)" stats --help 2>&1 \
  | tee "$OUT/nsys-stats-full-help.txt"
```

Find the CUDA API summary report.

A likely command is:

```bash
sudo "$(which nsys)" stats \
  --report cuda_api_sum \
  "$OUT/A-stock/roadNet-CA.nsys-rep" \
  | tee "$OUT/A-stock/roadNet-CA-cuda-api-summary.txt"
```

Look for:

```text
cudaMalloc
cudaMallocAsync
cudaFree
cudaFreeAsync
cudaMemcpy
cudaMemcpyAsync
cudaStreamSynchronize
cudaDeviceSynchronize
cudaLaunchKernel
```

Record:

```text
Count
Total Time
Average
Median
Min
Max
```

Repeat for `webbase-1M`.

---

# 11. GPU kernel summary

Find the CUDA kernel summary report name from `nsys stats --help`.

It is commonly similar to:

```text
cuda_gpu_kern_sum
```

Generate a kernel summary.

Identify:

```text
block_mapped_kernel
transform_reduce / reduction kernels
memset kernels if any
filter kernels if unexpectedly present
```

Record:

```text
number of block_mapped launches
number of reduction-related launches
```

---

# 12. Determine cudaMalloc/cudaFree periodicity

Do not rely on the summary alone.

We need to determine whether allocations occur:

```text
every BFS level
every second BFS level
only occasionally
only at frontier resize
or via another pattern
```

Obtain the CUDA API trace report if available.

Look for sequences approximately like:

```text
transform_reduce kernel(s)
cudaMalloc
block_mapped_kernel
cudaStreamSynchronize
cudaFree

transform_reduce kernel(s)
cudaMalloc
block_mapped_kernel
cudaStreamSynchronize
cudaFree
```

Count:

```text
N_block_mapped
N_cudaMalloc
N_cudaFree
```

Compute:

```text
mallocs_per_level =
    N_cudaMalloc / N_block_mapped

frees_per_level =
    N_cudaFree / N_block_mapped
```

Do not assume it is exactly 1.0.

The user specifically suspects a pattern approximately like:

```text
cudaMalloc/free appears once every two steps
```

Verify that hypothesis empirically.

---

# 13. Export Nsight Systems SQLite if needed

If `nsys stats` cannot expose temporal ordering clearly, export SQLite:

```bash
sudo "$(which nsys)" export \
  --type sqlite \
  --output "$OUT/A-stock/roadNet-CA.sqlite" \
  "$OUT/A-stock/roadNet-CA.nsys-rep"
```

Inspect schema:

```bash
sqlite3 "$OUT/A-stock/roadNet-CA.sqlite" '.tables'
```

Then:

```bash
sqlite3 "$OUT/A-stock/roadNet-CA.sqlite" \
  'SELECT name FROM sqlite_master WHERE type="table";'
```

Do not assume table names or schema.

Inspect candidates using:

```bash
sqlite3 "$OUT/A-stock/roadNet-CA.sqlite" \
  'PRAGMA table_info(TABLE_NAME);'
```

Identify:

- CUDA Runtime API activity;
- GPU kernel activity;
- String table.

Join the tables as necessary.

Generate a simplified chronological trace for at least 5-10 consecutive BFS levels:

```text
timestamp        duration        event
---------------------------------------------------
...              ...             transform_reduce
...              ...             cudaMalloc
...              ...             block_mapped_kernel
...              ...             cudaStreamSynchronize
...              ...             cudaFree
...
```

Repeat a smaller version for `webbase-1M` if its behavior differs.

---

# 14. Quantify allocation/free overhead

For the stock implementation, collect at least:

| Metric | roadNet-CA | webbase-1M |
|---|---:|---:|
| block_mapped launches | | |
| cudaMalloc calls | | |
| cudaFree calls | | |
| cudaMalloc total API time | | |
| cudaFree total API time | | |
| cudaStreamSynchronize calls | | |
| transform-reduce related kernels | | |
| GPU/native BFS time | | |

Also estimate:

```text
allocation_overhead =
    sum(cudaMalloc durations)
  + sum(cudaFree durations)
```

and approximately:

```text
allocation_fraction =
    allocation_overhead /
    BFS execution interval
```

Be careful:

`cudaMalloc` / `cudaFree` API durations may include implicit synchronization effects.

Therefore also inspect the GPU timeline:

```text
end of previous GPU work
        ↓
host/runtime gap
        ↓
start of next GPU work
```

The important quantity is not only API-call duration but also how much actual device idle time exists between useful kernels.

---

# 15. Save the stock binary

Before modifying source:

```bash
cp "$BFS_BIN" "$OUT/A-stock/bfs_bench.stock"

sha256sum "$BFS_BIN" \
  > "$OUT/A-stock/bfs_bench.stock.sha256"
```

This allows native A reruns even after rebuilding the working tree.

---

# 16. Create optimization branch

Only after completing the stock profiling:

```bash
git branch --show-current
git status --short
```

Create the branch **in place from the current HEAD**:

```bash
git checkout -b exp-opt
```

or:

```bash
git switch -c exp-opt
```

Confirm:

```bash
git branch --show-current
# exp-opt
```

Do not first checkout another branch or another commit.

---

# 17. Optimization structure

Do not implement everything in one change.

Use two stages.

## B0

Remove only:

```text
per-level block_offsets device allocation/free
```

Keep:

```text
compute_output_length()
transform_reduce
current synchronization
```

This isolates allocation/free cost.

## B

Starting from B0, additionally remove:

```text
compute_output_length()
transform_reduce degree sizing pre-pass
```

Use the persistent block counter to obtain the actual raw output length after the real advance kernel.

---

# 18. B0 implementation: persistent block counter only

## Objective

Replace:

```cpp
thrust::device_vector<typename frontier_t::offset_t>
    block_offsets(1);
```

with persistent device scratch already owned by the enactor.

Keep all other behavior identical.

---

# 19. Reuse `scanned_work_domain[0]`

The enactor already owns:

```cpp
thrust::device_vector<edge_t>
    scanned_work_domain;
```

For the `block_mapped` path, reuse:

```cpp
scanned_work_domain.data().get()
```

as the output counter.

Only element zero is needed.

The scratch storage is already persistent across BFS iterations.

---

# 20. Recommended B0 code structure

Prefer adding a dedicated enactor/scratch-aware block-mapped function instead of globally changing every direct API user.

For example:

```cpp
execute_with_scratch(...)
```

or:

```cpp
execute_preallocated(...)
```

Conceptually:

```cpp
template <
    advance_direction_t direction,
    advance_io_type_t input_type,
    advance_io_type_t output_type,
    typename graph_t,
    typename operator_t,
    typename frontier_t,
    typename work_tiles_t>
void execute_with_scratch(
    graph_t& G,
    operator_t op,
    frontier_t& input,
    frontier_t& output,
    work_tiles_t& scratch,
    gcuda::standard_context_t& context) {

  using offset_t =
      typename frontier_t::offset_t;

  // B0 keeps the original sizing pre-pass.
  auto size_of_output =
      compute_output_length(
          G,
          input,
          context);

  if (size_of_output <= 0) {
    output.set_number_of_elements(0);
    return;
  }

  if (output.get_capacity() <
      size_of_output) {
    output.reserve(size_of_output);
  }

  output.set_number_of_elements(
      size_of_output);

  if (scratch.size() < 1) {
    // Use Gunrock's normal error mechanism.
  }

  offset_t* d_block_offsets =
      scratch.data().get();

  error::throw_if_exception(
      hipMemsetAsync(
          d_block_offsets,
          0,
          sizeof(offset_t),
          context.stream()));

  // Use the exact same launch configuration.
  // Use the exact same block_mapped_kernel.
  // Do not modify the kernel algorithm.

  launch(
      ...,
      d_block_offsets);

  // Keep the stock synchronization in B0.
  context.synchronize();
}
```

Use the project's existing error-handling conventions.

Do not silently ignore CUDA/HIP API return values.

---

# 21. Dispatch B0 only through the enactor path

Relevant file:

```text
include/gunrock/framework/operators/advance/advance.hxx
```

The enactor already passes:

```cpp
E->scanned_work_domain
```

through its advance machinery.

For:

```cpp
load_balance_t::block_mapped
```

route the enactor-based path to the new persistent-scratch implementation.

Keep other paths unchanged:

```text
thread_mapped
merge_path
merge_path_v2
```

Do not modify BFS itself unless absolutely unavoidable.

The ideal B0 diff should touch only the advance framework.

---

# 22. Build B0

Do not rebuild the entire project unless required.

First inspect target names:

```bash
cmake --build build --target help \
  | grep -i bfs
```

Then rebuild only BFS:

```bash
cmake --build build \
  --target bfs_bench \
  -j"$(nproc)"
```

If the actual target name differs, use the target corresponding to:

```text
build/bin/bfs_bench
```

---

# 23. Validate B0

Use `roadNet-CA` first:

```bash
CUDA_VISIBLE_DEVICES=$TEST_GPU \
"$BFS_BIN" \
  --market "$GRAPH_ROAD" \
  --advance_load_balance block_mapped \
  --profile \
  --validate \
  | tee "$OUT/B0-persistent-counter/roadNet-CA-validate.txt"
```

Then `webbase-1M`:

```bash
CUDA_VISIBLE_DEVICES=$TEST_GPU \
"$BFS_BIN" \
  --market "$GRAPH_WEB" \
  --advance_load_balance block_mapped \
  --profile \
  --validate \
  | tee "$OUT/B0-persistent-counter/webbase-1M-validate.txt"
```

Both must be correct before profiling.

---

# 24. Profile B0

Use exactly the same Nsight Systems options used for A.

Example:

```bash
sudo env CUDA_VISIBLE_DEVICES=$TEST_GPU \
"$(which nsys)" profile \
  --trace=cuda,osrt,nvtx \
  --sample=none \
  --force-overwrite=true \
  -o "$OUT/B0-persistent-counter/roadNet-CA" \
  "$BFS_BIN" \
    --market "$GRAPH_ROAD" \
    --advance_load_balance block_mapped \
    --profile
```

Then `webbase-1M`.

Compare:

```text
cudaMalloc count
cudaFree count
cudaMalloc/free total time
transform_reduce kernel count
block_mapped kernel count
stream synchronization count
native runtime
```

Expected result:

```text
B0 should eliminate or substantially reduce
the per-level allocation/free pair caused by block_offsets.
```

Do not require all allocation API calls to disappear.

Thrust or other framework code may still allocate memory.

---

# 25. Commit B0

Inspect:

```bash
git diff
git status --short
```

Commit only source changes:

```bash
git add \
  include/gunrock/framework/operators/advance/block_mapped.hxx \
  include/gunrock/framework/operators/advance/advance.hxx
```

Then:

```bash
git commit -m \
  "opt: reuse persistent block-mapped output counter"
```

Do not commit profiler results.

---

# 26. Final B: remove the redundant sizing pre-pass

## Objective

Current stock/B0:

```text
degree transform_reduce
        ↓
host learns output size
        ↓
real advance
```

Final B:

```text
persistent counter = 0
        ↓
real block_mapped advance
        ↓
kernel-generated output count
        ↓
copy one scalar D2H
        ↓
one synchronization
        ↓
set host frontier length
```

This removes the first full frontier degree traversal.

---

# 27. Preconditions for final B

The optimized enactor path should require:

```text
context.size() == 1
scratch.size() >= 1
output capacity >= number_of_edges
```

The current BFS enactor should already satisfy the capacity condition due to its eager frontier reservation.

Still check explicitly.

If:

```cpp
output.get_capacity() <
    G.get_number_of_edges()
```

do not write beyond capacity.

Use a safe fallback to the stock block-mapped implementation.

---

# 28. Remove `compute_output_length()` in final B

In the optimized block-mapped enactor path, remove:

```cpp
auto size_of_output =
    compute_output_length(
        G,
        input,
        context);
```

Also remove the corresponding:

```cpp
output.reserve(size_of_output);
```

logic from this fast path.

Do not modify `compute_output_length()` globally.

Other operators or algorithms may still need it.

---

# 29. Reset the persistent counter

At the beginning of each advance:

```cpp
using offset_t =
    typename frontier_t::offset_t;

offset_t* d_output_count =
    scratch.data().get();

error::throw_if_exception(
    hipMemsetAsync(
        d_output_count,
        0,
        sizeof(offset_t),
        context.stream()));
```

Do not instantiate any temporary `thrust::device_vector`.

---

# 30. Keep the real block-mapped kernel unchanged

Do not modify:

```text
block_mapped_kernel
```

Specifically preserve:

```text
frontier vertex load
G.get_starting_edge(v)
G.get_number_of_neighbors(v)
CTA-local BlockScan
atomicAdd output positioning
shared-memory degree mapping
binary search
edge calculation
G.get_destination_vertex(e)
BFS search lambda
raw frontier output semantics
```

Only replace the counter pointer.

Stock:

```cpp
block_offsets.data().get()
```

Optimized:

```cpp
d_output_count
```

This is essential for a clean A/B experiment.

---

# 31. Obtain actual output size from the kernel counter

The existing kernel performs approximately:

```cpp
if (local_idx == 0)
    offset[0] =
        atomicAdd(
            block_offsets,
            aggregate_degree_per_block);
```

After all CTAs complete:

```text
block_offsets[0]
```

equals:

```text
sum of degrees of valid input-frontier vertices
```

which is the raw output frontier size.

After launching the kernel:

```cpp
offset_t h_output_count = 0;

error::throw_if_exception(
    hipMemcpyAsync(
        &h_output_count,
        d_output_count,
        sizeof(offset_t),
        hipMemcpyDeviceToHost,
        context.stream()));

context.synchronize();

output.set_number_of_elements(
    static_cast<std::size_t>(
        h_output_count));
```

Required ordering:

```text
block_mapped_kernel
        ↓
D2H output counter on same stream
        ↓
stream synchronization
        ↓
set_number_of_elements()
```

Do not read/set the count before synchronization.

---

# 32. Do not remove every synchronization

This experiment is not a full asynchronous enactor rewrite.

Current frontier length is host-managed metadata.

The next host loop needs a valid value for:

```text
is_converged()
input.get_number_of_elements()
```

Therefore one synchronization after the D2H counter copy is acceptable and expected.

Final B should reduce:

```text
pre-pass + synchronization/control dependency
+
advance synchronization
```

toward:

```text
one real advance
+
tiny D2H count
+
one synchronization
```

Do not attempt to build a fully GPU-resident convergence mechanism in this experiment.

---

# 33. Build final B

Rebuild only BFS:

```bash
cmake --build build \
  --target bfs_bench \
  -j"$(nproc)"
```

---

# 34. Validate final B

First `roadNet-CA`:

```bash
CUDA_VISIBLE_DEVICES=$TEST_GPU \
"$BFS_BIN" \
  --market "$GRAPH_ROAD" \
  --advance_load_balance block_mapped \
  --profile \
  --validate \
  | tee "$OUT/B-final/roadNet-CA-validate.txt"
```

Then `webbase-1M`:

```bash
CUDA_VISIBLE_DEVICES=$TEST_GPU \
"$BFS_BIN" \
  --market "$GRAPH_WEB" \
  --advance_load_balance block_mapped \
  --profile \
  --validate \
  | tee "$OUT/B-final/webbase-1M-validate.txt"
```

Only after both work correctly should `soc-LiveJournal1` be tested.

---

# 35. Profile final B

Profile `roadNet-CA` first:

```bash
sudo env CUDA_VISIBLE_DEVICES=$TEST_GPU \
"$(which nsys)" profile \
  --trace=cuda,osrt,nvtx \
  --sample=none \
  --force-overwrite=true \
  -o "$OUT/B-final/roadNet-CA" \
  "$BFS_BIN" \
    --market "$GRAPH_ROAD" \
    --advance_load_balance block_mapped \
    --profile
```

Then:

```bash
sudo env CUDA_VISIBLE_DEVICES=$TEST_GPU \
"$(which nsys)" profile \
  --trace=cuda,osrt,nvtx \
  --sample=none \
  --force-overwrite=true \
  -o "$OUT/B-final/webbase-1M" \
  "$BFS_BIN" \
    --market "$GRAPH_WEB" \
    --advance_load_balance block_mapped \
    --profile
```

Generate the same:

```text
CUDA API summary
CUDA kernel summary
chronological timeline
```

as for A.

---

# 36. Expected timeline differences

## A: stock

Expected structure:

```text
BFS level N

transform_reduce(degree)
        ↓
host/control dependency
        ↓
temporary block_offsets allocation
        ↓
block_mapped_kernel
        ↓
stream synchronization
        ↓
temporary block_offsets free
```

## B0

Expected:

```text
BFS level N

transform_reduce(degree)
        ↓
host/control dependency
        ↓
persistent counter memset
        ↓
block_mapped_kernel
        ↓
stream synchronization

no per-level block_offsets malloc/free
```

## B

Expected:

```text
BFS level N

persistent counter memset
        ↓
block_mapped_kernel
        ↓
tiny D2H output-count copy
        ↓
single stream synchronization
```

---

# 37. Native performance methodology

Do not use profiler runtime as the final performance number.

Nsight Systems is for explaining the timeline.

Native execution determines actual speedup.

All native runs must include:

```bash
--profile
```

---

# 38. Development-stage native testing

During optimization/debugging, use only:

```text
roadNet-CA
webbase-1M
```

Start with approximately 5 warm-up/debug runs if needed.

Do not repeatedly run `soc-LiveJournal1` during development.

---

# 39. Final native timing runs

For each implementation:

```text
A
B0
B
```

and each main dataset:

```text
roadNet-CA
webbase-1M
```

perform at least:

```text
3 warm-up runs
20 measured runs
```

Example:

```bash
for i in $(seq 1 3); do
  CUDA_VISIBLE_DEVICES=$TEST_GPU \
  "$BFS_BIN" \
    --market "$GRAPH_ROAD" \
    --advance_load_balance block_mapped \
    --profile \
    >/dev/null
done
```

Measured:

```bash
for i in $(seq 1 20); do
  CUDA_VISIBLE_DEVICES=$TEST_GPU \
  "$BFS_BIN" \
    --market "$GRAPH_ROAD" \
    --advance_load_balance block_mapped \
    --profile \
    | tee -a "$OUT/B-final/roadNet-CA-native-runs.txt"
done
```

Repeat for:

```text
webbase-1M
```

and for A/B0/B.

Use the saved stock binary when necessary:

```bash
"$OUT/A-stock/bfs_bench.stock"
```

---

# 40. Final soc-LiveJournal1 evaluation

Only after:

```text
B validates correctly
roadNet-CA results look reasonable
webbase-1M results look reasonable
Nsight confirms the expected structural changes
```

run:

```bash
CUDA_VISIBLE_DEVICES=$TEST_GPU \
"$BFS_BIN" \
  --market "$GRAPH_LJ" \
  --advance_load_balance block_mapped \
  --profile \
  --validate
```

If validation is prohibitively expensive, validate correctness on the two smaller datasets and perform native-only evaluation on LiveJournal, but explicitly state that decision in the report.

For final performance evaluation, fewer repetitions may be acceptable because LiveJournal is relatively slow.

Recommended:

```text
2 warm-up runs
5-10 measured runs
```

rather than 20.

Example:

```bash
for i in $(seq 1 2); do
  CUDA_VISIBLE_DEVICES=$TEST_GPU \
  "$BFS_BIN" \
    --market "$GRAPH_LJ" \
    --advance_load_balance block_mapped \
    --profile \
    >/dev/null
done
```

Then:

```bash
for i in $(seq 1 8); do
  CUDA_VISIBLE_DEVICES=$TEST_GPU \
  "$BFS_BIN" \
    --market "$GRAPH_LJ" \
    --advance_load_balance block_mapped \
    --profile \
    | tee -a "$OUT/B-final/soc-LiveJournal1-native-runs.txt"
done
```

Perform the same number of repetitions for A and B.

B0 on LiveJournal is optional unless attribution remains unclear after the smaller datasets.

---

# 41. Statistics

For every dataset/version calculate:

```text
mean
median
standard deviation
minimum
maximum
```

Prefer median as the primary comparison.

Calculate:

```text
speedup_B0 =
    median(A) /
    median(B0)

speedup_B =
    median(A) /
    median(B)
```

And isolate contributions:

```text
malloc_only_gain =
    median(A) -
    median(B0)

prepass_removal_gain =
    median(B0) -
    median(B)
```

---

# 42. Main result tables

Produce one table per dataset.

Example:

| Metric | A stock | B0 persistent counter | B final |
|---|---:|---:|---:|
| Correct | yes | yes | yes |
| BFS levels / block_mapped launches | | | |
| cudaMalloc calls | | | |
| cudaFree calls | | | |
| cudaMalloc total API time | | | |
| cudaFree total API time | | | |
| transform-reduce kernels | | | |
| stream/device sync calls | | | |
| median native BFS time | | | |
| speedup vs A | 1.00x | | |

Produce this for:

```text
roadNet-CA
webbase-1M
soc-LiveJournal1
```

For LiveJournal, B0 may be omitted if runtime cost is excessive.

---

# 43. How to interpret the results

## Case 1: A -> B0 produces most of the gain

Conclusion:

```text
per-level allocation/free is a major
software-engineering overhead.
```

## Case 2: B0 -> B produces most of the gain

Conclusion:

```text
the redundant degree/output-size sizing pass
and its host-control dependency dominate
the avoidable framework overhead.
```

## Case 3: B produces only small improvement

Conclusion:

```text
the allocation/pre-pass overhead is visible
but actual graph traversal and memory behavior
still dominate native runtime.
```

This is still a useful result.

Do not exaggerate the significance of allocator activity merely because it looks large under profiling.

---

# 44. Research interpretation: producer-to-consumer gap

This experiment matters for the prefetch study.

## Stock A

The observed gap may contain:

```text
row-offset / degree metadata known
        ↓
transform_reduce sizing pass
        ↓
host-control dependency
        ↓
temporary allocation/free
        ↓
real block_mapped kernel
        ↓
actual column_indices[e] load
```

A large fraction of this may be purely software/framework-generated headroom.

## Final B

The relevant sequence should become approximately:

```text
block_mapped_kernel

frontier vertex v
        ↓
get_starting_edge(v)
get_number_of_neighbors(v)
        ↓
CTA-local BlockScan
        ↓
atomic block output positioning
        ↓
binary search
        ↓
edge ID
        ↓
column_indices[e]
```

This is much closer to the actual algorithmic producer-consumer gap.

The final report must explicitly separate:

```text
Software/framework-created gap:
    redundant degree pre-pass
    temporary allocation/free
    host synchronization/control

Algorithmic gap:
    degree/row-start production
    CTA block scan
    global atomic output allocation
    binary search
    edge calculation
    adjacency load
```

The B version should be the preferred baseline for later prefetch-headroom analysis.

---

# 45. Optional Nsight Compute verification

If B shows a substantial speedup, verify that the actual `block_mapped_kernel` did not accidentally change.

Use Nsight Systems first to find the exact kernel name.

Then profile one representative invocation with:

```bash
sudo env CUDA_VISIBLE_DEVICES=$TEST_GPU \
"$(which ncu)" \
  --target-processes application-only \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  --kernel-name 'regex:.*block_mapped.*' \
  ...
```

The BFS command must still include:

```bash
--profile
```

Do not profile every BFS level with NCU.

Use:

```text
--launch-skip
--launch-count
```

to select one representative middle/frontier-heavy launch.

A and B should have similar:

```text
instructions
occupancy
memory behavior
branch behavior
kernel-level work
```

because the kernel itself should remain unchanged.

---

# 46. Optional filter sensitivity experiment

Only after the main experiment is complete, optionally test:

```bash
--enable_filter
--filter_algorithm predicated
```

Run:

```text
A + filter
B + filter
```

This can show whether INVALID frontier entries amplify the next-level overhead.

Do not mix filter-enabled results into the main filter-disabled A/B comparison.

---

# 47. Code review before finalizing

Run:

```bash
git diff exp-env..exp-opt -- \
  include/gunrock/framework/operators/advance/block_mapped.hxx \
  include/gunrock/framework/operators/advance/advance.hxx
```

The desired final state is:

```text
BFS algorithm file:
    unchanged

block_mapped_kernel:
    unchanged

changed:
    output counter lifetime
    output-size determination
    synchronization organization
```

Do not change:

```text
CSR representation
distance semantics
neighbor traversal
CTA scan
binary search
load-balancing mapping
```

Check:

```bash
git status --short
git log --oneline --decorate -5
```

Recommended commits:

```text
opt: reuse persistent block-mapped output counter

opt: remove redundant block-mapped sizing prepass
```

---

# 48. Important failure modes

## Allocation pattern differs from expectation

If `cudaMalloc` / `cudaFree` does not occur once per level, do not force the hypothesis.

Possible causes include:

```text
Thrust allocator caching
frontier resizing
allocation every second iteration
other framework allocations
cudaMallocAsync instead of cudaMalloc
HIP compatibility wrappers
```

Report the actual observed periodicity.

---

## Final B produces invalid memory access

Check:

```text
output.get_capacity()
```

against:

```text
G.get_number_of_edges()
```

If insufficient, fall back to stock sizing logic.

Never allow the kernel to write past capacity.

---

## Validation fails

Temporarily compare:

```text
stock size_of_output
optimized d_output_count
```

for the same iteration.

Useful debug information:

```text
iteration
input frontier length
stock output size
optimized output counter
```

Remove or disable debug prints before performance runs.

Check:

```text
counter reset
counter type width
same CUDA stream
D2H synchronization ordering
set_number_of_elements timing
```

---

## Nsight does not show cudaMalloc/free

Confirm:

```text
--trace=cuda
```

and search for:

```text
cudaMalloc
cudaMallocAsync
cudaFree
cudaFreeAsync
cuMemAlloc
cuMemFree
```

depending on runtime/library behavior.

---

## Profiler shows a large gain but native runtime does not

Trust native measurements.

Nsight Systems can perturb:

```text
runtime APIs
kernel launch costs
synchronization
```

Use:

```text
Nsight for attribution
native runs for final performance
```

---

# 49. Final report structure

Produce a Markdown report with the following sections.

## Environment

Include:

```text
GPU
CUDA version
GCC version
Nsight Systems version
Nsight Compute version
Git commit
branch
```

## Stock code path

Show:

```text
degree transform_reduce
        ↓
allocation
        ↓
block_mapped
        ↓
synchronization
        ↓
free
```

with source filenames.

## Stock timeline evidence

For both:

```text
roadNet-CA
webbase-1M
```

show a representative sequence covering several BFS levels.

Explicitly answer:

```text
Does cudaMalloc/cudaFree occur periodically?
How frequently?
How much time does it consume?
Does the GPU become idle around those operations?
```

## B0 result

Explain:

```text
which allocation/free calls disappeared
native runtime change
remaining transform_reduce behavior
```

## B result

Explain:

```text
whether the degree pre-pass disappeared
whether only one real traversal remains
remaining D2H/synchronization behavior
```

## Native performance

Report:

```text
roadNet-CA
webbase-1M
soc-LiveJournal1
```

with A/B0/B as appropriate.

## Research interpretation

Explicitly separate:

```text
framework-induced gap
```

from:

```text
algorithmic producer-consumer gap
```

and answer:

> After eliminating avoidable Gunrock software-engineering overhead, is there still meaningful time between obtaining CSR adjacency metadata and consuming `column_indices[e]`?

This conclusion is more important than the raw optimization speedup.

---

# 50. Final experiment philosophy

This task is **not**:

> Make Gunrock BFS as fast as possible.

It is:

> Remove clearly avoidable framework overhead without changing the BFS or block-mapped load-balancing algorithm, then determine what producer-consumer separation remains in the real GPU traversal.

Use:

```text
roadNet-CA
webbase-1M
```

for development and detailed profiling.

Use:

```text
soc-LiveJournal1
```

only after the implementation is stable, as a final larger-scale evaluation workload.
