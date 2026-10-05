#pragma once

#include <atomic>
#include <chrono>
#include <vector>
#include <omp.h>
#include <thrust/host_vector.h>

namespace bc_cpu {

/// Parallel, level-synchronous single-source Brandes reference. Each level
/// finishes before its successor starts; backward reads are therefore stable.
/// Match Gunrock's 0.5 scaling and exclusion of the source vertex.
template <typename csr_t, typename vertex_t, typename weight_t>
float run(const csr_t& csr, vertex_t source, weight_t* values) {
  using edge_t = typename csr_t::offset_type;
  thrust::host_vector<edge_t> offsets(csr.row_offsets);
  thrust::host_vector<vertex_t> neighbors(csr.column_indices);
  const auto n = csr.number_of_rows;
  std::vector<std::atomic<vertex_t>> labels(n);
  std::vector<double> sigmas(n, 0.0);
  std::vector<double> deltas(n, 0.0);
  std::vector<std::vector<vertex_t>> levels;

#pragma omp parallel for
  for (vertex_t v = 0; v < n; ++v) {
    labels[v].store(-1, std::memory_order_relaxed);
    values[v] = 0;
  }
  labels[source].store(0, std::memory_order_relaxed);
  sigmas[source] = 1.0;
  levels.push_back({source});
  const auto start = std::chrono::steady_clock::now();

  for (vertex_t depth = 0; !levels.back().empty(); ++depth) {
    const auto& frontier = levels.back();
    std::vector<vertex_t> next;
#pragma omp parallel
    {
      std::vector<vertex_t> local;
#pragma omp for schedule(dynamic, 64)
      for (std::size_t i = 0; i < frontier.size(); ++i) {
        const vertex_t src = frontier[i];
        for (edge_t e = offsets[src]; e < offsets[src + 1]; ++e) {
          const vertex_t dst = neighbors[e];
          vertex_t expected = -1;
          if (labels[dst].compare_exchange_strong(expected, depth + 1,
                                                  std::memory_order_relaxed))
            local.push_back(dst);
          if (labels[dst].load(std::memory_order_relaxed) == depth + 1) {
#pragma omp atomic update
            sigmas[dst] += sigmas[src];
          }
        }
      }
#pragma omp critical
      next.insert(next.end(), local.begin(), local.end());
    }
    if (next.empty())
      break;
    levels.push_back(std::move(next));
  }

  for (int depth = static_cast<int>(levels.size()) - 1; depth > 0; --depth) {
    const auto& frontier = levels[depth];
#pragma omp parallel for schedule(dynamic, 64)
    for (std::size_t i = 0; i < frontier.size(); ++i) {
      const vertex_t src = frontier[i];
      double delta = 0.0;
      for (edge_t e = offsets[src]; e < offsets[src + 1]; ++e) {
        const vertex_t dst = neighbors[e];
        if (labels[dst].load(std::memory_order_relaxed) == depth + 1)
          delta += sigmas[src] / sigmas[dst] * (1.0 + deltas[dst]);
      }
      deltas[src] = delta;
      values[src] = static_cast<weight_t>(0.5 * delta);
    }
  }
  return std::chrono::duration<float, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace bc_cpu
