#pragma once

#include <gunrock/framework/operators/advance/swpf/types.hxx>

namespace gunrock::operators::advance::swpf {

template <>
struct swpf_policy<swpf_algorithm_t::gp> {
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
                                             state_prefetch_t state_prefetch) {
    edge_record_t<graph_t> records[distance];
    while (true) {
      int count = 0;
      // next() skips row boundaries: every occupied slot is an actual edge.
#pragma unroll
      for (int i = 0; i < distance; ++i) {
        if (!next(records[i]))
          break;
        prepare<target>(graph, records[i]);
        ++count;
      }
      if (count == 0)
        break;
#pragma unroll
      for (int i = 0; i < distance; ++i)
        if (i < count)
          prefetch<target>(graph, records[i], state_prefetch);
#pragma unroll
      for (int i = 0; i < distance; ++i)
        if (i < count)
          consume(records[i]);
    }
  }
};

}  // namespace gunrock::operators::advance::swpf
