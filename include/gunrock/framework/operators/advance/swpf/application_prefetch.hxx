#pragma once

#include <gunrock/framework/operators/advance/swpf/prefetch_ptx.hxx>

namespace gunrock::operators::advance::swpf {

struct no_state_prefetch_t {
  // Generic advance operators may consume edge weights.
  static constexpr bool uses_edge_weights = true;

  template <swpf_target_t target, typename vertex_t>
  __device__ __forceinline__ void operator()(vertex_t, vertex_t) const {}
};

template <typename distance_t, bool read_source = false>
struct distance_prefetch_t {
  // BFS only updates destination depth; SSSP reads source distance and weight.
  static constexpr bool uses_edge_weights = read_source;

  distance_t* distances;

  template <swpf_target_t target, typename vertex_t>
  __device__ __forceinline__ void operator()(vertex_t src, vertex_t dst) const {
    // SSSP source distances can also be atomic targets of other edges.
    if constexpr (read_source)
      prefetch_atomic_target(distances + src);
    prefetch_atomic_target(distances + dst);
  }
};

template <typename vertex_t, typename weight_t>
struct bc_forward_prefetch_t {
  static constexpr bool uses_edge_weights = false;

  vertex_t* labels;
  weight_t* sigmas;

  template <swpf_target_t target>
  __device__ __forceinline__ void operator()(vertex_t src, vertex_t dst) const {
    prefetch_read<target>(labels + src);
    prefetch_read<target>(sigmas + src);
    prefetch_atomic_target(labels + dst);
    prefetch_atomic_target(sigmas + dst);
  }
};

template <typename vertex_t, typename weight_t>
struct bc_backward_prefetch_t {
  static constexpr bool uses_edge_weights = false;

  vertex_t* labels;
  weight_t* sigmas;
  weight_t* deltas;
  weight_t* bc_values;

  template <swpf_target_t target>
  __device__ __forceinline__ void operator()(vertex_t src, vertex_t dst) const {
    prefetch_read<target>(labels + src);
    prefetch_read<target>(sigmas + src);
    prefetch_read<target>(labels + dst);
    prefetch_read<target>(sigmas + dst);
    prefetch_read<target>(deltas + dst);
    prefetch_atomic_target(deltas + src);
    prefetch_atomic_target(bc_values + src);
  }
};

}  // namespace gunrock::operators::advance::swpf
