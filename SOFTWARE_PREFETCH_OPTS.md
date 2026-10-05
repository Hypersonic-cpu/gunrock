# Gunrock V2 Merge-Path Software Prefetch Implementation Guide

## Goal

Implement modular software prefetch support for Gunrock V2 `merge_path` advance, targeting:

- BFS
- SSSP
- BC forward
- BC backward

The implementation must support runtime selection:

```bash
--swpf=none
--swpf=gp
--swpf=spp
```

It must also support selecting the prefetch cache target:

```bash
--swpf-target=l1
--swpf-target=l2
```

`none` must preserve the original behavior and performance path as closely as possible.

Do **not** modify unrelated load-balancing algorithms.

---

## 1. Relevant execution path

Focus on:

```text
include/gunrock/framework/operators/advance/merge_path.hxx
```

The important inner path is approximately:

```cpp
v = input.get_element_at(global_row);
starting_edge = G.get_starting_edge(v);
e = starting_edge + rank;

n = G.get_destination_vertex(e);
w = G.get_edge_weight(e);

op(v, n, e, w);
```

For BFS / SSSP / BC, the important dependent chain is:

```text
edge e
  ↓
destination[e]
  ↓
application_state[destination]
  ↓
op(...)
```

Software prefetch should pipeline this chain.

---

## 2. Configuration

Add common enums, preferably under the existing operator/configuration layer:

```cpp
enum class swpf_algorithm_t {
    none,
    gp,
    spp
};

enum class swpf_target_t {
    l1,
    l2
};
```

Parse:

```bash
--swpf=none
--swpf=gp
--swpf=spp

--swpf-target=l1
--swpf-target=l2

--swpf-distance=N
```

Recommended defaults:

```text
--swpf=none
--swpf-target=l2
--swpf-distance=2
```

The option must flow cleanly:

```text
benchmark CLI
    ↓
algorithm options
    ↓
advance::execute(...)
    ↓
merge_path::execute(...)
    ↓
merge_path_kernel(...)
```

Avoid global variables.

---

## 3. Modular structure

Do not scatter prefetch logic throughout `merge_path.hxx`.

Create something similar to:

```text
include/gunrock/framework/operators/advance/swpf/
    swpf.hxx
    prefetch_ptx.hxx
    none.hxx
    group_prefetch.hxx
    software_pipeline.hxx
```

Provide a common policy interface:

```cpp
template <swpf_algorithm_t algorithm>
struct swpf_policy;
```

Conceptually:

```cpp
swpf_policy<...>::prepare(...);
swpf_policy<...>::prefetch(...);
swpf_policy<...>::consume(...);
```

Keep merge-path scheduling independent from application-specific prefetch behavior.

---

## 4. Prefetch primitive: NVIDIA PTX

The software-prefetch baseline must use NVIDIA PTX cache-prefetch instructions:

```ptx
prefetch.global.L1 [addr];
prefetch.global.L2 [addr];
```

Provide reusable device wrappers in:

```text
include/gunrock/framework/operators/advance/swpf/prefetch_ptx.hxx
```

For example:

```cpp
__device__ __forceinline__
void prefetch_l1(const void* ptr) {
  asm volatile(
      "prefetch.global.L1 [%0];"
      :
      : "l"(ptr)
  );
}

__device__ __forceinline__
void prefetch_l2(const void* ptr) {
  asm volatile(
      "prefetch.global.L2 [%0];"
      :
      : "l"(ptr)
  );
}
```

Do not embed inline PTX repeatedly in application code.

Do **not** use:

- dummy loads as the primary prefetch mechanism;
- `cp.async`;
- shared-memory copies as a replacement for the original global access.

`cp.async` is global-to-shared data movement and is not the cache-prefetch baseline intended here.

For the initial implementation, do not add eviction-policy modifiers such as `::evict_last`.

---

## 5. Prefetch target semantics

`--swpf-target` controls where ordinary prefetched data should be brought.

### `--swpf-target=l2`

All explicit software prefetches use:

```ptx
prefetch.global.L2
```

Example:

```text
destination[e]
edge_weight[e]
distances[dst]
labels[dst]
sigmas[dst]
deltas[dst]
    ↓
prefetch.global.L2
```

### `--swpf-target=l1`

Ordinary read-only/read-mostly accesses should use:

```ptx
prefetch.global.L1
```

Examples include:

```text
destination[e]
edge_weight[e]
labels[dst]      // when only read
sigmas[dst]      // when only read
deltas[dst]      // when only read
```

However, **addresses that will subsequently be accessed by a global atomic operation must still be prefetched to L2**, even when:

```text
--swpf-target=l1
```

Examples:

```text
BFS:
    atomicMin(&distances[dst], ...)
    → prefetch.global.L2 &distances[dst]

SSSP:
    atomicMin(&distances[dst], ...)
    → prefetch.global.L2 &distances[dst]

BC forward:
    atomicCAS(&labels[dst], ...)
    atomicAdd(&sigmas[dst], ...)
    → prefetch.global.L2 for these targets

BC backward:
    atomicAdd(&deltas[src], ...)
    atomicAdd(&bc_values[src], ...)
    → atomic targets remain L2-prefetched if explicitly prefetched
```

Therefore the policy is:

```text
target=l1:
    normal loads  → L1
    atomic target → L2

target=l2:
    normal loads  → L2
    atomic target → L2
```

Implement helper functions so this rule is centralized rather than duplicated across BFS/SSSP/BC.

For example:

```cpp
template <swpf_target_t target>
__device__ __forceinline__
void prefetch_read(const void* ptr);

__device__ __forceinline__
void prefetch_atomic_target(const void* ptr) {
  prefetch_l2(ptr);
}
```

The real load or atomic operation must remain unchanged after the prefetch.

---

## 6. `none`

`none` is the reference implementation.

It should compile to essentially the existing code:

```cpp
n = G.get_destination_vertex(e);
w = G.get_edge_weight(e);
op(v, n, e, w);
```

Use compile-time specialization where practical so that:

```text
--swpf=none
```

introduces no explicit PTX prefetch and minimal control overhead.

---

## 7. GP: Group Prefetch

Implement a simple baseline first.

For each thread, collect a small group of upcoming **actual edges**:

```text
e0 e1 e2 e3
```

Then:

```text
prefetch/read destination[e0..e3]
prefetch/read weight[e0..e3]

once destination is known:
    prefetch application_state[dst]

process e0..e3
```

All explicit cache prefetches must go through the common PTX wrapper and obey `--swpf-target`.

Important:

> A merge-path loop iteration is not necessarily an edge.

The serial merge can advance either an edge or a row boundary.

Therefore GP must operate on the next `K` **real edge operations**, not simply `item + K`.

---

## 8. SPP: Software-Pipelined Prefetch

SPP is the primary implementation.

Implement a pipeline similar to:

```text
Stage 1:
    obtain future edge e

Stage 2:
    load/prefetch destination[e]
    load/prefetch weight[e]

Stage 3:
    destination is known
    prefetch application state indexed by destination

Stage 4:
    execute op()
```

Conceptually:

```text
edge i+2   → destination
edge i+1   → application_state[destination]
edge i     → op()
```

All cache prefetches obey `--swpf-target`, except atomic targets which always use L2.

Use:

```bash
--swpf-distance=N
```

to control lookahead.

Do not assume the next merge-path iteration corresponds to the next edge.

---

## 9. Application-specific prefetch targets

### BFS

Critical operation:

```cpp
atomicMin(&distances[neighbor], iteration + 1);
```

Dependency:

```text
edge
 → neighbor
 → distances[neighbor]
 → atomicMin
```

After the future `neighbor` becomes known:

```cpp
prefetch_atomic_target(&distances[neighbor]);
```

This always means L2, regardless of `--swpf-target`.

Do not replace the atomic with a shared-memory copy.

---

### SSSP

Important accesses:

```cpp
source_distance = distances[source];
weight = edge_weight[e];
atomicMin(&distances[neighbor], source_distance + weight);
```

Prefetch/load ahead:

```text
destination[e]
edge_weight[e]
```

These follow `--swpf-target`.

Once destination is known:

```cpp
prefetch_atomic_target(&distances[destination]);
```

which always targets L2.

`distances[source]` should preferably be loaded once and reused when multiple edges share the same source.

---

### BC Forward

Important accesses:

```cpp
labels[src]
labels[dst]
sigmas[src]
sigmas[dst]
```

The current operations on destination state include atomics, so destination atomic targets should use L2:

```cpp
prefetch_atomic_target(&labels[dst]);
prefetch_atomic_target(&sigmas[dst]);
```

Source-side read-only values may follow `--swpf-target`.

---

### BC Backward

Important destination-side accesses:

```cpp
labels[dst]
sigmas[dst]
deltas[dst]
```

Dependency:

```text
edge
  ↓
dst
  ├── labels[dst]
  ├── sigmas[dst]
  └── deltas[dst]
```

These are reads in the backward operator and should follow `--swpf-target`:

```cpp
prefetch_read<target>(&labels[dst]);
prefetch_read<target>(&sigmas[dst]);
prefetch_read<target>(&deltas[dst]);
```

Atomic updates such as:

```cpp
atomicAdd(&deltas[src], ...);
atomicAdd(&bc_values[src], ...);
```

must remain unchanged; if explicitly prefetched, their targets must use L2.

---

## 10. Register pressure

SPP will require temporary state such as:

```cpp
future_e
future_dst
future_weight
future_valid
```

Keep the number of simultaneously buffered edges small.

Start with:

```text
N = 1, 2, 4
```

Do not build a large AMAC-style state machine initially.

Measure register count and occupancy.

---

## 11. Required CLI behavior

Examples:

```bash
./bfs ... --advance=merge_path \
    --swpf=none

./bfs ... --advance=merge_path \
    --swpf=gp \
    --swpf-target=l1 \
    --swpf-distance=4

./bfs ... --advance=merge_path \
    --swpf=spp \
    --swpf-target=l2 \
    --swpf-distance=2
```

Same interface for:

```text
bfs
sssp
bc
```

Semantics:

```text
--swpf=none
    no explicit PTX prefetch

--swpf=gp
    grouped PTX software prefetch

--swpf=spp
    software-pipelined PTX prefetch

--swpf-target=l1
    ordinary reads → L1
    atomic targets → L2

--swpf-target=l2
    all prefetches → L2
```

Invalid values must produce a clear error.

---

## 12. Correctness tests

For every application compare:

```text
none
gp
spp
```

against the original implementation.

Required:

```text
BFS:
    identical distance array

SSSP:
    identical distances within normal floating-point tolerance

BC:
    identical / numerically equivalent BC values
```

Test at least:

```text
roadNet-CA
soc-orkut
Indochina-2004
```

---

## 13. Performance evaluation

For each application report:

```text
GPU kernel time
speedup over --swpf=none
registers/thread
occupancy
L1 hit rate
L2 hit rate
DRAM bytes
long-scoreboard stalls
eligible warps/cycle
```

Sweep:

```text
swpf   = none, gp, spp
target = l1, l2
distance = 1, 2, 4, 8
```

For `swpf=none`, `target` has no effect.

Do not claim success solely from higher cache hit rate. The primary metric is kernel execution time.

Also verify that improvements are not caused merely by changed occupancy or register allocation.

---

## 14. Implementation order

Implement in this order:

```text
1. CLI/config plumbing
2. reusable PTX prefetch_l1()/prefetch_l2() wrappers
3. centralized read-vs-atomic target policy
4. swpf=none
5. GP
6. SPP edge → dst pipeline
7. BFS destination-state prefetch
8. SSSP destination-state prefetch
9. BC forward
10. BC backward
11. correctness tests
12. target/distance performance sweep
```

Keep every stage compiling and runnable before proceeding.

The final code should make it easy to add another software-prefetch policy or cache target without modifying the core merge-path scheduling logic.
