/**
 * @file block_mapped.cuh
 * @brief Unit tests for enactor-based block-mapped advance dispatch.
 */

#include <gunrock/cuda/context.hxx>
#include <gunrock/framework/enactor.hxx>
#include <gunrock/framework/operators/advance/advance.hxx>
#include <gunrock/graph/graph.hxx>
#include <gunrock/io/sample.hxx>

#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <memory>

struct block_mapped_test_accept_all_t {
  __host__ __device__ bool operator()(const int&, const int&, const int&,
                                      const float&) const {
    return true;
  }
};

struct block_mapped_test_count_edges_t {
  int* count;

  __device__ bool operator()(const int&,
                             const int&,
                             const int&,
                             const float&) const {
    atomicAdd(count, 1);
    return true;
  }
};

template <typename graph_t>
struct block_mapped_test_problem_t : gunrock::problem_t<graph_t> {
  using base_t = gunrock::problem_t<graph_t>;
  using base_t::base_t;

  void init() override {}
  void reset() override {}
};

template <typename graph_t,
          gunrock::operators::advance::advance_output_bound_t output_bound =
              gunrock::operators::advance::advance_output_bound_t::unknown>
struct block_mapped_test_enactor_t
    : gunrock::enactor_t<block_mapped_test_problem_t<graph_t>> {
  using problem_t = block_mapped_test_problem_t<graph_t>;
  using base_t = gunrock::enactor_t<problem_t>;
  using base_t::base_t;

  static constexpr gunrock::operators::advance::advance_output_bound_t
      advance_output_bound = output_bound;

  void loop(gunrock::gcuda::multi_context_t&) override {}
};

TEST(operators_advance, block_mapped_unknown_bound_uses_sized_output) {
  auto csr = gunrock::io::sample::csr();
  gunrock::graph::graph_properties_t properties;
  auto G = gunrock::graph::build<gunrock::memory_space_t::device>(properties,
                                                                 csr);
  using graph_t = decltype(G);
  using bound_t = gunrock::operators::advance::advance_output_bound_t;
  auto context = std::make_shared<gunrock::gcuda::multi_context_t>(0);
  block_mapped_test_problem_t<graph_t> problem(G, context);
  block_mapped_test_enactor_t<graph_t, bound_t::unknown> enactor(&problem,
                                                                 context);

  enactor.get_input_frontier()->push_back(1);
  gunrock::operators::advance::execute<
      gunrock::operators::load_balance_t::block_mapped,
      gunrock::operators::advance_direction_t::forward,
      gunrock::operators::advance_io_type_t::vertices,
      gunrock::operators::advance_io_type_t::vertices>(
      G, &enactor, block_mapped_test_accept_all_t{}, *context, false);

  auto* output = enactor.get_output_frontier();
  ASSERT_EQ(output->get_number_of_elements(), 2);
  thrust::host_vector<int> output_host(output->get_number_of_elements());
  thrust::copy_n(thrust::device_pointer_cast(output->data()),
                 output->get_number_of_elements(), output_host.begin());
  std::sort(output_host.begin(), output_host.end());
  EXPECT_EQ(output_host[0], 0);
  EXPECT_EQ(output_host[1], 1);
}

TEST(operators_advance, block_mapped_declared_bound_uses_preallocated_output) {
  auto csr = gunrock::io::sample::csr();
  gunrock::graph::graph_properties_t properties;
  auto G = gunrock::graph::build<gunrock::memory_space_t::device>(properties,
                                                                 csr);
  using graph_t = decltype(G);
  using bound_t = gunrock::operators::advance::advance_output_bound_t;
  auto context = std::make_shared<gunrock::gcuda::multi_context_t>(0);
  block_mapped_test_problem_t<graph_t> problem(G, context);
  block_mapped_test_enactor_t<graph_t, bound_t::graph_edges> enactor(&problem,
                                                                      context);

  enactor.get_input_frontier()->push_back(1);
  gunrock::operators::advance::execute<
      gunrock::operators::load_balance_t::block_mapped,
      gunrock::operators::advance_direction_t::forward,
      gunrock::operators::advance_io_type_t::vertices,
      gunrock::operators::advance_io_type_t::vertices>(
      G, &enactor, block_mapped_test_accept_all_t{}, *context, false);

  EXPECT_EQ(enactor.get_output_frontier()->get_number_of_elements(), 2);
}

TEST(operators_advance, block_mapped_graph_input_uses_preallocated_output) {
  auto csr = gunrock::io::sample::csr();
  gunrock::graph::graph_properties_t properties;
  auto G = gunrock::graph::build<gunrock::memory_space_t::device>(properties,
                                                                 csr);
  using graph_t = decltype(G);
  using bound_t = gunrock::operators::advance::advance_output_bound_t;
  auto context = std::make_shared<gunrock::gcuda::multi_context_t>(0);
  block_mapped_test_problem_t<graph_t> problem(G, context);
  block_mapped_test_enactor_t<graph_t, bound_t::unknown> enactor(&problem,
                                                                 context);

  gunrock::operators::advance::execute<
      gunrock::operators::load_balance_t::block_mapped,
      gunrock::operators::advance_direction_t::forward,
      gunrock::operators::advance_io_type_t::graph,
      gunrock::operators::advance_io_type_t::vertices>(
      G, &enactor, block_mapped_test_accept_all_t{}, *context, false);

  auto* output = enactor.get_output_frontier();
  ASSERT_EQ(output->get_number_of_elements(), 4);
  thrust::host_vector<int> output_host(output->get_number_of_elements());
  thrust::copy_n(thrust::device_pointer_cast(output->data()),
                 output->get_number_of_elements(), output_host.begin());
  std::sort(output_host.begin(), output_host.end());
  EXPECT_EQ(output_host[0], 0);
  EXPECT_EQ(output_host[1], 1);
  EXPECT_EQ(output_host[2], 1);
  EXPECT_EQ(output_host[3], 2);
}

TEST(operators_advance, block_mapped_graph_input_capacity_falls_back_safely) {
  auto csr = gunrock::io::sample::csr();
  gunrock::graph::graph_properties_t properties;
  auto G = gunrock::graph::build<gunrock::memory_space_t::device>(properties,
                                                                 csr);
  using graph_t = decltype(G);
  using bound_t = gunrock::operators::advance::advance_output_bound_t;
  auto context = std::make_shared<gunrock::gcuda::multi_context_t>(0);
  block_mapped_test_problem_t<graph_t> problem(G, context);
  gunrock::enactor_properties_t enactor_properties;
  enactor_properties.self_manage_frontiers = true;
  block_mapped_test_enactor_t<graph_t, bound_t::unknown> enactor(
      &problem, context, enactor_properties);

  ASSERT_EQ(enactor.get_output_frontier()->get_capacity(), 0);
  gunrock::operators::advance::execute<
      gunrock::operators::load_balance_t::block_mapped,
      gunrock::operators::advance_direction_t::forward,
      gunrock::operators::advance_io_type_t::graph,
      gunrock::operators::advance_io_type_t::vertices>(
      G, &enactor, block_mapped_test_accept_all_t{}, *context, false);

  EXPECT_GE(enactor.get_output_frontier()->get_capacity(),
            G.get_number_of_edges());
  EXPECT_EQ(enactor.get_output_frontier()->get_number_of_elements(), 4);
}

TEST(operators_advance, block_mapped_no_output_skips_output_management) {
  auto csr = gunrock::io::sample::csr();
  gunrock::graph::graph_properties_t properties;
  auto G = gunrock::graph::build<gunrock::memory_space_t::device>(properties,
                                                                 csr);
  using graph_t = decltype(G);
  using bound_t = gunrock::operators::advance::advance_output_bound_t;
  auto context = std::make_shared<gunrock::gcuda::multi_context_t>(0);
  block_mapped_test_problem_t<graph_t> problem(G, context);
  block_mapped_test_enactor_t<graph_t, bound_t::unknown> enactor(&problem,
                                                                 context);
  thrust::device_vector<int> device_count(1, 0);

  gunrock::operators::advance::execute<
      gunrock::operators::load_balance_t::block_mapped,
      gunrock::operators::advance_direction_t::forward,
      gunrock::operators::advance_io_type_t::graph,
      gunrock::operators::advance_io_type_t::none>(
      G, &enactor,
      block_mapped_test_count_edges_t{device_count.data().get()}, *context,
      false);

  thrust::host_vector<int> count_host(device_count);
  EXPECT_EQ(count_host[0], 4);
}
