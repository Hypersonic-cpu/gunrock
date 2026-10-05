#pragma once

#include <gunrock/framework/operators/advance/swpf/types.hxx>

namespace gunrock::operators::advance::swpf {

template <>
struct swpf_policy<swpf_algorithm_t::none> {
  template <int distance,
            int max_edges,
            swpf_target_t target,
            typename graph_t,
            typename next_t,
            typename consume_t,
            typename state_prefetch_t>
  __device__ __forceinline__ static void run(const graph_t& graph,
                                             next_t next,
                                             consume_t consume,
                                             state_prefetch_t) {
    edge_record_t<graph_t> record;
    while (next(record)) {
      record.destination = graph.get_destination_vertex(record.edge);
      record.weight = graph.get_edge_weight(record.edge);
      consume(record);
    }
  }
};

}  // namespace gunrock::operators::advance::swpf
