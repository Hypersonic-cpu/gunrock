# Gunrock `block_mapped` Safety Refactor and Re-Profiling Plan

Machine-specific checkout, dataset, result, GPU, and profiler values are
defined in `AGENTS.local.md`. Set the variables documented there before using
the commands in this plan; this document keeps only portable repository and
algorithm guidance.

## Objective

Continue from the existing Gunrock repository and the current optimization branch:

```bash
exp-opt
```

Do **not** start from a clean checkout, do not discard the existing B0 / Final-B work, and do not redesign the block-mapped load-balancing algorithm.

The purpose of this task is to turn the current optimization into a clean, generic software-engineering fix:

1. Keep the safe generic fix that removes repeated per-iteration temporary `cudaMalloc` / `cudaFree`.
2. Keep the more aggressive no-sizing optimization where it can be **proven safe**.
3. Avoid hard-coding special cases such as `if (algorithm == BFS)`.
4. Express the safety condition using C++17 compile-time traits/capabilities.
5. Automatically use Final-B when:
   - `output_type == none`, or
   - `input_type == graph`, or
   - an enactor explicitly declares that its raw block-mapped output is bounded by the graph edge count.
6. Use the generic B0 path for all other block-mapped calls until their output-size invariant is proven.
7. Rebuild, validate correctness, and re-run Nsight Systems / optional Nsight Compute profiling.
8. Confirm that BFS performance remains approximately unchanged from the previous Final-B results.

Previous measured BFS medians:

```text
Graph               Stock       B0          Final B
-----------------------------------------------------
roadNet-CA           137.847 ms  90.186 ms   13.362 ms
webbase-1M             2.195 ms   2.051 ms    1.809 ms
soc-LiveJournal1      23.865 ms      —       16.447 ms
```

The new refactor should preserve approximately the same Final-B performance while making the implementation safe and reusable.

---

# 1. Important scope

This task is **not** about improving the `block_mapped` algorithm.

Do not change:

- the block-mapped kernel algorithm;
- CTA-local `BlockScan`;
- source-vertex binary search;
- CSR edge traversal;
- BFS `atomicMin` semantics;
- graph representation;
- load-balancing policy;
- filter semantics;
- traversal ordering.

We are only fixing framework/software-engineering behavior around:

```text
output sizing
temporary allocation
persistent scratch
frontier-length publication
compile-time safety selection
```

---

# 2. C++ language constraint

Gunrock current main is built with:

```text
C++17
CUDA C++17
HIP C++17
```

Therefore:

- **Do not use C++20 concepts.**
- Prefer:
  - `static constexpr`
  - traits
  - `if constexpr`
  - `static_assert`
  - `std::is_same_v`
  - `std::remove_cv_t`
  - `std::void_t` only if necessary

Avoid complicated SFINAE unless there is no simpler C++17 solution.

The preferred design is a simple semantic capability trait.

---

# 3. First inspect the current branch and diff

Run:

```bash
git branch --show-current
git status --short
git log --oneline --decorate -8
git diff exp-env
```

The branch should be:

```text
exp-opt
```

Important existing modified files include:

```text
include/gunrock/framework/operators/advance/advance.hxx
include/gunrock/framework/operators/advance/block_mapped.hxx
include/gunrock/compat/runtime_api.h
```

Also inspect:

```text
include/gunrock/framework/enactor.hxx
include/gunrock/algorithms/bfs.hxx
include/gunrock/algorithms/sssp.hxx
include/gunrock/algorithms/ppr.hxx
include/gunrock/algorithms/kcore.hxx
include/gunrock/algorithms/tc.hxx
include/gunrock/algorithms/hits.hxx
include/gunrock/algorithms/spmv.hxx
```

Do not discard the existing optimization commits.

---

# 4. Existing B0 and Final-B behavior

The current optimization has two conceptual stages.

## B0 — generic persistent-counter fix

The original stock implementation performs:

```cpp
compute_output_length(...);

thrust::device_vector<offset_t> block_offsets(1);

block_mapped_kernel(...);

context.synchronize();
```

B0 removes the repeated temporary counter allocation:

```text
compute_output_length()
        ↓
persistent counter reset
        ↓
block_mapped_kernel
        ↓
synchronize
```

This is a generic safe fix because output sizing is still performed exactly as in stock Gunrock.

---

## Final B — skip the redundant sizing pre-pass

Final B performs:

```text
persistent counter reset
        ↓
block_mapped_kernel
        ↓
copy actual output count D2H
        ↓
synchronize
        ↓
publish output frontier length
```

It removes:

```cpp
compute_output_length(...)
```

and therefore also removes the redundant degree `transform_reduce` traversal.

However, this is only safe when the already allocated output frontier is guaranteed to be large enough for the complete raw adjacency expansion.

That guarantee must now be represented explicitly.

---

# 5. Do not hard-code BFS in `advance.hxx`

Do **not** implement logic such as:

```cpp
if constexpr (is_bfs<enactor_type>) {
    ...
}
```

or:

```cpp
if (algorithm_name == "BFS") {
    ...
}
```

The framework should not care about algorithm names.

Instead, express the actual semantic property:

> Is the complete raw output expansion of this advance guaranteed not to exceed the graph edge count?

This is the property that matters for using a graph-edge-sized preallocated output frontier.

---

# 6. Introduce a semantic output-bound capability

Add a small compile-time enum, preferably in an advance-related configuration/header where it does not create circular includes.

For example:

```cpp
enum class advance_output_bound_t {
  unknown,
  graph_edges
};
```

Meaning:

```text
unknown:
    The framework cannot prove a bound from compile-time semantics.
    Use the safe B0 sizing path.

graph_edges:
    The caller guarantees that the complete raw output expansion
    cannot exceed G.get_number_of_edges().
    Final-B no-sizing is allowed when capacity is sufficient.
```

Do not encode algorithm names in this enum.

---

# 7. Add a default capability to the enactor

Prefer adding a default capability to the common base enactor:

```text
include/gunrock/framework/enactor.hxx
```

For example:

```cpp
static constexpr auto advance_output_bound =
    operators::advance::advance_output_bound_t::unknown;
```

If placing the enum directly in `operators::advance` creates include-order problems, place the enum in an appropriate lower-level config namespace/header and reference it consistently.

The important behavior is:

```text
all enactors default to unknown
```

No algorithm should silently receive Final-B behavior unless the framework can prove it from the I/O type or the algorithm explicitly declares the capability.

---

# 8. BFS should declare the capability, not be special-cased

In the BFS enactor:

```text
include/gunrock/algorithms/bfs.hxx
```

declare:

```cpp
static constexpr auto advance_output_bound =
    operators::advance::advance_output_bound_t::graph_edges;
```

or the equivalent namespace/type chosen above.

Why BFS is safe:

```text
The valid BFS output frontier represents newly discovered vertices.

The BFS search lambda uses atomicMin, so only a successful first discovery
returns true for a vertex in a given traversal.

Therefore valid BFS frontier entries are bounded by graph vertices, and the
sum of adjacency degrees of those valid frontier vertices cannot exceed the
total CSR edge count.

Thus a graph-edge-sized output frontier is sufficient for the complete raw
block-mapped expansion.
```

Do not describe this as a generic frontier invariant.

This declaration is a semantic promise made by BFS.

---

# 9. Automatically recognize `input_type == graph`

Any block-mapped advance with:

```cpp
input_type == advance_io_type_t::graph
```

traverses the graph domain itself rather than an arbitrary frontier.

For forward vertex traversal, each graph vertex is represented once in the input domain.

Therefore the total raw adjacency expansion is bounded by the graph edge count.

The framework should automatically recognize this compile-time property.

Do **not** require HITS, SpMV, or other graph-input algorithms to manually opt in.

---

# 10. Automatically recognize `output_type == none`

For:

```cpp
output_type == advance_io_type_t::none
```

there is no output frontier to size.

Therefore:

```text
compute_output_length()
output capacity checking
output counter reset
D2H output count
frontier-length publication
```

are unnecessary.

This case should bypass output-sizing logic entirely.

Example:

```text
SpMV:
input_type  = graph
output_type = none
```

Do not perform a useless `hipMemsetAsync` of the output counter merely because the block-mapped kernel signature contains the pointer.

The kernel's compile-time branch should not dereference it when `output_type == none`.

---

# 11. Define one compile-time safety predicate

Implement a clear C++17 compile-time predicate in the generic enactor-based advance dispatch.

Conceptually:

```cpp
constexpr bool output_not_required =
    output_type == advance_io_type_t::none;

constexpr bool graph_input_bound =
    input_type == advance_io_type_t::graph;

constexpr bool declared_graph_edge_bound =
    enactor_type::advance_output_bound ==
        advance_output_bound_t::graph_edges;

constexpr bool can_skip_sizing =
    output_not_required ||
    graph_input_bound ||
    declared_graph_edge_bound;
```

Use `if constexpr`.

Do not use runtime branches for compile-time I/O properties.

---

# 12. Desired generic dispatch behavior

The final block-mapped enactor dispatch should behave conceptually like:

```cpp
if constexpr (lb == load_balance_t::block_mapped) {

  if constexpr (output_type == advance_io_type_t::none) {

    // No output sizing is necessary.
    block_mapped::execute_no_output(...);

  } else if constexpr (
      input_type == advance_io_type_t::graph ||
      enactor_type::advance_output_bound ==
          advance_output_bound_t::graph_edges) {

    // Final-B path.
    block_mapped::execute_preallocated(...);

  } else {

    // Generic B0 path.
    block_mapped::execute_with_scratch(...);
  }

} else {

  // Existing Gunrock path for other load balancers.
  ...
}
```

Exact function names are flexible, but the semantic distinction must be clear.

Recommended names:

```text
execute_with_scratch()
    = generic B0, keeps compute_output_length()

execute_preallocated()
    = Final B, skips compute_output_length()

execute_no_output()
    = optional helper for output_type == none
```

It is also acceptable to fold `output_type == none` into one helper using `if constexpr`, as long as no unnecessary sizing/reset/D2H work remains.

---

# 13. Generic B0 path

The generic path must preserve stock Gunrock output-sizing semantics.

Conceptually:

```cpp
template <...>
void execute_with_scratch(...) {

  if constexpr (output_type != advance_io_type_t::none) {

    auto size_of_output =
        compute_output_length(G, input, context);

    if (size_of_output <= 0) {
      output.set_number_of_elements(0);
      return;
    }

    if (output.get_capacity() < size_of_output)
      output.reserve(size_of_output);

    output.set_number_of_elements(size_of_output);
  }

  // Use persistent scratch instead of device_vector(1).

  ...
}
```

The only important software change relative to stock is:

```text
temporary block counter
    ↓
persistent block counter
```

Do not remove the sizing pre-pass from this path.

This is the safe default for:

```text
SSSP
PPR
KCore
other arbitrary frontier-producing algorithms
```

until their output bounds are explicitly proven.

---

# 14. Final-B preallocated path

The Final-B path must skip:

```cpp
compute_output_length(...)
```

For output-producing advances:

1. Check that output capacity is sufficient.
2. Reset persistent counter.
3. Launch the unchanged block-mapped kernel.
4. Copy the actual raw output count D2H.
5. Synchronize.
6. Publish the host-side frontier length.

Conceptually:

```cpp
using offset_counter_t =
    typename work_tiles_t::value_type;

auto* d_output_count =
    scratch.data().get();

if (output.get_capacity() <
    static_cast<std::size_t>(
        G.get_number_of_edges())) {

  // Safety fallback.
  execute_with_scratch(...);
  return;
}

error::throw_if_exception(
    hipMemsetAsync(
        d_output_count,
        0,
        sizeof(offset_counter_t),
        context.stream()));

launch unchanged block_mapped_kernel(...);

offset_counter_t h_output_count = 0;

error::throw_if_exception(
    hipMemcpyAsync(
        &h_output_count,
        d_output_count,
        sizeof(offset_counter_t),
        hipMemcpyDeviceToHost,
        context.stream()));

context.synchronize();

output.set_number_of_elements(
    static_cast<std::size_t>(
        h_output_count));
```

The fallback must remain.

A compile-time semantic guarantee does not imply that a caller actually reserved enough memory.

---

# 15. Add compile-time safety guards

Use `static_assert` where appropriate.

For example, in the persistent-counter implementation:

```cpp
static_assert(
    std::is_same_v<
        std::remove_cv_t<offset_counter_t>,
        std::remove_cv_t<typename frontier_t::offset_t>>,
    "Persistent block-mapped counter type must match frontier offset type");
```

For the no-sizing helper, if the helper has access to `enactor_type`, add a semantic assertion:

```cpp
static_assert(
    input_type == advance_io_type_t::graph ||
    output_type == advance_io_type_t::none ||
    enactor_type::advance_output_bound ==
        advance_output_bound_t::graph_edges,
    "No-sizing block-mapped advance requires a proven output bound");
```

If the low-level helper does not know `enactor_type`, keep this `static_assert` at the higher-level dispatch wrapper.

Do not add a meaningless assert that only repeats a runtime capacity check.

The compile-time assert should protect the semantic capability; the runtime check should protect actual allocated capacity.

---

# 16. Reuse persistent scratch cleanly

The current experiment reuses:

```cpp
E->scanned_work_domain[0]
```

as the persistent block counter.

For this task, that is acceptable.

Add a clear comment such as:

```text
block_mapped does not consume the global scanned work domain;
element 0 is reused as a persistent output counter to avoid
per-advance device allocation.
```

Requirements:

1. Ensure `scratch.size() >= 1`.
2. Add the type `static_assert`.
3. Do not assume the remaining scratch contents are preserved.
4. Ensure no nested/concurrent operation aliases this scratch at the same time.
5. Do not introduce a new per-level allocation.

Do not perform a large framework redesign merely to rename this scratch buffer.

---

# 17. Preserve the original block-mapped kernel

The body of:

```cpp
block_mapped_kernel(...)
```

must remain unchanged.

Specifically preserve:

```text
input frontier load
G.get_starting_edge(v)
G.get_number_of_neighbors(v)
CTA-local CUB BlockScan
global atomicAdd for block output offsets
shared-memory degree table
binary search / upper_bound
edge ID calculation
G.get_destination_vertex(e)
user-provided lambda
raw output write
```

After editing, verify this explicitly with Git diff.

---

# 18. Do not remove the required D2H frontier-length dependency

For Final-B frontier-producing calls, keep:

```text
kernel
  ↓
small D2H counter copy
  ↓
context.synchronize()
  ↓
set_number_of_elements()
```

Do not attempt to remove this synchronization in this task.

The current Gunrock enactor uses host-managed frontier length for:

```text
next iteration input size
convergence
frontier buffer management
```

This remaining synchronization is not the same problem as the removed redundant degree-sizing pre-pass.

---

# 19. Expected automatic behavior after refactor

The final compile-time selection should approximately be:

| Case | Path |
|---|---|
| `output_type == none` | no output sizing |
| `input_type == graph`, output exists | Final B automatically |
| Enactor declares `graph_edges` output bound | Final B automatically |
| Arbitrary frontier-producing block-mapped call | B0 |
| Non-block-mapped load balancer | unchanged Gunrock behavior |

Examples:

```text
BFS:
    input_type = vertices
    BFS declares graph_edges bound
    → Final B

HITS graph traversal:
    input_type = graph
    → Final B automatically

SpMV:
    input_type = graph
    output_type = none
    → no sizing and no output-counter overhead

SSSP:
    input_type = vertices
    no proven output bound yet
    → B0

PPR:
    input_type = vertices
    no proven output bound yet
    → B0

KCore:
    input_type = vertices
    no proven output bound yet
    → B0
```

Do not add capabilities to SSSP/PPR/KCore merely because they might work empirically.

Only declare a capability after proving the raw expansion bound.

---

# 20. Inspect all current block-mapped callers

Search the repository:

```bash
grep -RIn \
  "load_balance_t::block_mapped\|execute_runtime" \
  include/gunrock/algorithms
```

At minimum inspect:

```text
BFS
SSSP
PPR
KCore
TC
HITS
SpMV
```

For each caller, record:

```text
input_type
output_type
whether it declares an output-bound capability
selected path: no-output / Final-B / B0
```

Add a short table to the final report.

---

# 21. Build all affected targets

Discover targets:

```bash
cmake --build build --target help \
  | grep -Ei \
    'bfs|sssp|ppr|kcore|hits|spmv|tc'
```

Build every available affected target.

At minimum:

```text
bfs_bench
bfs
sssp
spmv
```

and any available PPR/KCore/HITS/TC targets.

Use:

```bash
cmake --build build \
  --target <target> \
  -j"$(nproc)"
```

Do not do a destructive clean rebuild unless necessary.

---

# 22. BFS correctness validation

Use the standalone `bfs` example for correctness because `bfs_bench` has different argument handling.

Validate at least:

```text
roadNet-CA
webbase-1M
```

with source 0.

Expected:

```text
Number of errors: 0
```

Do not perform expensive LiveJournal CPU validation unless practical.

---

# 23. Non-BFS correctness / smoke tests

Test the algorithms affected by generic block-mapped dispatch:

```text
SSSP
PPR
KCore
TC
HITS
SpMV
```

For each:

1. Build successfully.
2. Run on a small/medium graph.
3. Use built-in validation/reference checking if available.
4. Ensure no:
   - illegal memory access;
   - assertion;
   - crash;
   - obviously invalid result.
5. Record which dispatch path is expected:
   - B0;
   - Final B because `input_type == graph`;
   - no-output path.

Give special attention to:

```text
SSSP:
    should remain B0

HITS:
    graph input should automatically use Final B

SpMV:
    graph input + output none should avoid sizing entirely
```

---

# 24. Verify no-sizing selection without benchmark-name hacks

Inspect the final diff and ensure there is no logic equivalent to:

```cpp
is_bfs
algorithm == bfs
bfs-specific branch inside advance.hxx
```

The only BFS-specific code should be its capability declaration:

```cpp
advance_output_bound = graph_edges;
```

The actual dispatch must be generic.

---

# 25. BFS native performance re-evaluation

All `bfs_bench` runs must use:

```bash
--profile
```

This is required to avoid interference from Gunrock/NVBench built-in profiling behavior in this experiment.

Executable:

```bash
"$REPO_ROOT/build/bin/bfs_bench"
```

Datasets:

```bash
export GRAPH_ROAD="$DATA_ROOT/roadNet-CA.mtx"
export GRAPH_WEB="$DATA_ROOT/webbase-1M.mtx"
export GRAPH_LJ="$DATA_ROOT/soc-LiveJournal1.mtx"
```

Use the same physical GPU identified by `AGENTS.local.md` for all runs.

Run in this order:

```text
1. roadNet-CA
2. webbase-1M
3. soc-LiveJournal1 only after the first two look correct
```

For:

```text
roadNet-CA
webbase-1M
```

use:

```text
3 warmups
20 measured runs
```

For:

```text
soc-LiveJournal1
```

use:

```text
2 warmups
8 measured runs
```

Canonical command:

```bash
CUDA_VISIBLE_DEVICES=$TEST_GPU \
  "$REPO_ROOT/build/bin/bfs_bench" \
  --market "$GRAPH_ROAD" \
  --profile
```

Do not pass unsupported options to `bfs_bench`.

---

# 26. Expected BFS performance

Previous Final-B medians:

```text
roadNet-CA:
    13.361664 ms

webbase-1M:
    1.808896 ms

soc-LiveJournal1:
    16.447488 ms
```

The trait/capability refactor should not materially change the BFS hot path.

As a sanity check:

```text
roadNet-CA:
    preferably within ~10%

soc-LiveJournal1:
    preferably within ~10%

webbase-1M:
    may be noisier because the workload is very short
```

If roadNet or LiveJournal changes substantially, investigate before accepting the refactor.

---

# 27. Mandatory Nsight Systems re-profile

Use:

```bash
sudo -n "$(which nsys)"
```

For profiler runs:

```bash
sudo env CUDA_VISIBLE_DEVICES=$TEST_GPU \
  "$(which nsys)" profile ...
```

Every `bfs_bench` invocation must still contain:

```bash
--profile
```

Profile at least:

```text
roadNet-CA
webbase-1M
```

and then:

```text
soc-LiveJournal1
```

after the first two are verified.

Use the same Nsight Systems options as the previous experiment whenever possible.

---

# 28. Nsight expectations for BFS Final-B

The previous roadNet Final-B profile approximately showed:

```text
block_mapped launches:
    556

sizing-reduction kernels:
    0

cudaMalloc/cudaFree:
    13 / 13 setup-level calls

small D2H frontier-count copies:
    ~556

counter resets:
    ~557

aggregate block_mapped kernel time:
    ~2.96 ms
```

After the trait/capability refactor:

```text
sizing-reduction kernels must remain zero

per-level cudaMalloc/free must remain eliminated

small D2H counter publication should remain

block_mapped kernel body must remain unchanged
```

Exact counts may differ slightly.

Do not require exact equality.

---

# 29. Mandatory non-BFS Nsight Systems check

Profile at least one non-BFS frontier-producing algorithm that remains on B0.

Prefer:

```text
SSSP
```

The expected B0 structure is:

```text
compute_output_length / degree sizing
        ↓
persistent counter reset
        ↓
block_mapped kernel
        ↓
synchronization
```

Important expectations:

```text
sizing reduction remains present

per-iteration temporary block counter allocation/free is gone
```

This proves that the generic allocator fix remains active without applying an unsafe no-sizing assumption.

---

# 30. Profile one `input_type == graph` application

Profile at least one application whose block-mapped input type is `graph`.

Prefer one already known to use block-mapped, such as HITS if its executable is available.

Verify:

```text
input_type == graph
        ↓
automatic Final-B selection
        ↓
no compute_output_length sizing reduction
        ↓
no per-iteration temporary block counter allocation
```

Do not add an algorithm-specific trait for this case.

The automatic compile-time `input_type == graph` rule should be sufficient.

---

# 31. Verify `output_type == none`

Use SpMV if practical.

Confirm:

```text
input_type = graph
output_type = none
```

Expected software path:

```text
no output sizing
no output frontier reserve
no output counter reset needed
no output counter D2H
```

The actual block-mapped traversal kernel should still run normally.

Use Nsight Systems if necessary to confirm there is no redundant output-management activity.

---

# 32. Optional Nsight Compute sanity check

Nsight Systems is mandatory.

If practical, use Nsight Compute on one representative `block_mapped_kernel` launch.

Use:

```bash
sudo -n "$(which ncu)"
```

Select one kernel with:

```text
--launch-skip
--launch-count
```

Useful sections:

```text
SpeedOfLight
MemoryWorkloadAnalysis
WarpStateStats
SchedulerStats
```

The purpose is only to verify that the actual kernel behavior did not materially change.

Do not profile every BFS level.

---

# 33. Final code-review requirements

Run:

```bash
git diff exp-env
git status --short
git log --oneline --decorate -10
```

Verify:

### Must remain unchanged

```text
block_mapped_kernel algorithm
BFS atomicMin semantics
CSR representation
load-balancing mapping
binary search
neighbor traversal
```

### Must now be generic

```text
Final-B eligibility selection
```

based on:

```text
output_type == none
input_type == graph
declared output-bound capability
```

### Must remain safe by default

```text
unknown frontier semantics
    → B0
```

### Must not exist

```text
algorithm-name checks
BFS-specific branch in generic advance dispatch
```

---

# 34. Keep CUDA compatibility aliases

Retain the compatibility additions if still required:

```cpp
#define hipMemcpyAsync cudaMemcpyAsync
#define hipMemsetAsync cudaMemsetAsync
```

They are compatibility plumbing, not algorithmic changes.

Ensure the AMD/HIP side is not accidentally broken by these changes.

---

# 35. Suggested commit structure

Prefer one focused refactor commit after validation:

```bash
git commit -m \
  "refactor: select block-mapped no-sizing path by output bounds"
```

If implementation naturally splits, acceptable commits are:

```text
refactor: add block-mapped output-bound capability

refactor: select block-mapped sizing path at compile time
```

Do not commit profiling artifacts.

Keep:

```text
profiling-results/
```

ignored.

---

# 36. Update the experiment report

Update:

```text
"$RESULTS_ROOT/native-bfs-opt/REPORT.md"
```

Add a section:

```text
Output-bound capability refactor
```

Explain:

```text
B0:
    generic safe persistent-counter path

Final B:
    compile-time-selected no-sizing path

Automatic Final-B conditions:
    output_type == none
    input_type == graph
    enactor declares graph_edges bound

BFS:
    declares graph_edges capability

SSSP/PPR/KCore:
    remain B0 until proven safe
```

---

# 37. Required final result tables

## BFS performance

| Graph | Previous Final B | Refactored Final B | Change |
|---|---:|---:|---:|
| roadNet-CA | 13.362 ms | | |
| webbase-1M | 1.809 ms | | |
| soc-LiveJournal1 | 16.447 ms | | |

---

## BFS Nsight structure

| Metric | Previous Final B | Refactored Final B |
|---|---:|---:|
| block_mapped launches | | |
| sizing reduction kernels | 0 | |
| cudaMalloc calls | ~13 | |
| cudaFree calls | ~13 | |
| D2H output-count copies | | |
| GPU interval | | |

---

## Algorithm-path classification

| Algorithm | Input type | Output type | Capability | Expected path |
|---|---|---|---|---|
| BFS | vertices | vertices | graph_edges | Final B |
| SSSP | vertices | vertices | unknown | B0 |
| PPR | vertices | vertices | unknown | B0 |
| KCore | vertices | vertices | unknown | B0 |
| HITS | graph | vertices | automatic | Final B |
| SpMV | graph | none | automatic | no-output |
| TC | inspect | inspect | inspect | derive from semantics |

Fill this table from the actual source, not assumptions.

---

# 38. Required final conclusions

The final report must answer all of the following explicitly:

1. Did BFS performance remain approximately unchanged after removing the benchmark-specific behavior?
2. Did BFS still have zero sizing-reduction kernels?
3. Did per-level `cudaMalloc/cudaFree` remain eliminated?
4. Does generic frontier-producing block-mapped execution now preserve stock sizing semantics?
5. Does SSSP or another B0 algorithm run correctly?
6. Does at least one `input_type == graph` application automatically take the Final-B path?
7. Does `output_type == none` avoid unnecessary sizing/counter work?
8. Is the block-mapped kernel itself unchanged?
9. Are there any algorithms for which the new compile-time trait changes correctness or performance unexpectedly?

Do not claim completion until correctness checks and profiling support the conclusions.

---

# 39. Design principle

The final implementation should follow this rule:

```text
Do not ask:
    "Which algorithm is this?"

Ask:
    "Can the framework prove that this advance does not need dynamic output sizing?"
```

The resulting structure should be:

```text
block_mapped
    │
    ├── output_type == none
    │      └── no output sizing
    │
    ├── input_type == graph
    │      └── Final B automatically
    │
    ├── caller declares graph_edges output bound
    │      └── Final B automatically
    │
    └── otherwise
           └── B0 safe path
```

This is the desired software-engineering end state.
