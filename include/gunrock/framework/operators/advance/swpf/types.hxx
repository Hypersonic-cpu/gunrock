#pragma once

#include <gunrock/framework/operators/advance/swpf/application_prefetch.hxx>

namespace gunrock::operators::advance::swpf {

template <swpf_algorithm_t algorithm>
struct swpf_policy;

template <typename graph_t>
struct edge_record_t {
  typename graph_t::vertex_type source;
  typename graph_t::edge_type edge;
  int output_index;
  typename graph_t::vertex_type destination;
  typename graph_t::weight_type weight;
};

template <swpf_target_t target, typename graph_t>
__device__ __forceinline__ void prepare(const graph_t& graph,
                                        const edge_record_t<graph_t>& record) {
  prefetch_read<target>(graph.get_column_indices() + record.edge);
  prefetch_read<target>(graph.get_nonzero_values() + record.edge);
}

template <swpf_target_t target, typename graph_t, typename state_prefetch_t>
__device__ __forceinline__ void prefetch(const graph_t& graph,
                                         edge_record_t<graph_t>& record,
                                         state_prefetch_t state_prefetch) {
  record.destination = graph.get_destination_vertex(record.edge);
  record.weight = graph.get_edge_weight(record.edge);
  state_prefetch.template operator()<target>(record.source, record.destination);
}

}  // namespace gunrock::operators::advance::swpf
