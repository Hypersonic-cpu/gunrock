// Compile-only regression: unweighted hooks must work with no weight accessors.
#include <gunrock/framework/operators/advance/swpf/types.hxx>

namespace swpf = gunrock::operators::advance::swpf;
using gunrock::operators::swpf_target_t;

struct unweighted_graph_t {
  using vertex_type = int;
  using edge_type = int;
  using weight_type = float;

  int* columns;

  __device__ int* get_column_indices() const { return columns; }
  __device__ int get_destination_vertex(int edge) const { return columns[edge]; }
  __device__ float* get_nonzero_values() const = delete;
  __device__ float get_edge_weight(int) const = delete;
};

using bfs_hook_t = swpf::distance_prefetch_t<int>;
using sssp_hook_t = swpf::distance_prefetch_t<float, true>;
using bc_forward_hook_t = swpf::bc_forward_prefetch_t<int, float>;
using bc_backward_hook_t = swpf::bc_backward_prefetch_t<int, float>;

static_assert(!bfs_hook_t::uses_edge_weights);
static_assert(sssp_hook_t::uses_edge_weights);
static_assert(!bc_forward_hook_t::uses_edge_weights);
static_assert(!bc_backward_hook_t::uses_edge_weights);
static_assert(swpf::no_state_prefetch_t::uses_edge_weights);

template <typename hook_t>
__device__ void check_hook(unweighted_graph_t graph, hook_t hook) {
  swpf::edge_record_t<unweighted_graph_t> record{};
  swpf::prepare<swpf_target_t::l1, hook_t>(graph, record);
  swpf::prefetch<swpf_target_t::l1>(graph, record, hook);
  swpf::prepare<swpf_target_t::l2, hook_t>(graph, record);
  swpf::prefetch<swpf_target_t::l2>(graph, record, hook);
}

__global__ void check_unweighted_hooks(int* columns,
                                      int* labels,
                                      float* sigmas,
                                      float* deltas,
                                      float* bc_values) {
  unweighted_graph_t graph{columns};
  check_hook(graph, bfs_hook_t{labels});
  check_hook(graph, bc_forward_hook_t{labels, sigmas});
  check_hook(graph, bc_backward_hook_t{labels, sigmas, deltas, bc_values});
}
