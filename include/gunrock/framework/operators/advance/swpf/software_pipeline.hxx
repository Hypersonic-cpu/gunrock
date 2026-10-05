#pragma once

#include <gunrock/framework/operators/advance/swpf/types.hxx>

namespace gunrock::operators::advance::swpf {

template <>
struct swpf_policy<swpf_algorithm_t::spp> {
  // Constant array indices allow nvcc to scalarize the small ring into
  // registers instead of placing dynamically indexed records on the stack.
  template <typename record_t, int capacity>
  __device__ __forceinline__ static record_t read_slot(
      const record_t (&records)[capacity],
      int slot) {
    record_t record{};
#pragma unroll
    for (int i = 0; i < capacity; ++i)
      if (i == slot)
        record = records[i];
    return record;
  }

  template <typename record_t, int capacity>
  __device__ __forceinline__ static void
  write_slot(record_t (&records)[capacity], int slot, const record_t& record) {
#pragma unroll
    for (int i = 0; i < capacity; ++i)
      if (i == slot)
        records[i] = record;
  }

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
    // A thread owns at most max_edges actual edges, including pipeline drain.
    constexpr int capacity =
        (2 * distance + 1 < max_edges) ? 2 * distance + 1 : max_edges;
    edge_record_t<graph_t> records[capacity];
    int produced = 0;
    bool exhausted = false;
    for (int step = 0; !exhausted || step < produced + 2 * distance; ++step) {
      // Stage 1: obtain a future edge and prefetch its immutable CSR data.
      if (!exhausted) {
        edge_record_t<graph_t> record{};
        if (next(record)) {
          prepare<target>(graph, record);
          write_slot(records, step % capacity, record);
          ++produced;
        } else {
          exhausted = true;
        }
      }
      // Stage 2: resolve destination, then prefetch dependent application
      // state.
      const int load_index = step - distance;
      if (load_index >= 0 && load_index < produced) {
        auto record = read_slot(records, load_index % capacity);
        prefetch<target>(graph, record, state_prefetch);
        write_slot(records, load_index % capacity, record);
      }
      // Stage 3: preserve the original edge order and execute the original op.
      const int consume_index = step - 2 * distance;
      if (consume_index >= 0 && consume_index < produced)
        consume(read_slot(records, consume_index % capacity));
    }
  }
};

}  // namespace gunrock::operators::advance::swpf
