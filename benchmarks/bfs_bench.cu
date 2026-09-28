#include <nvbench/nvbench.cuh>
#include <cxxopts.hpp>
#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/algorithms/bfs.hxx>

#include "benchmarks.hxx"
#include "profiling_range.hxx"

using namespace gunrock;
using namespace memory;

using vertex_t = int;
using edge_t = int;
using weight_t = float;

std::string filename;
int profile_repetitions = 1;

struct parameters_t {
  std::string filename;
  bool help = false;
  cxxopts::Options options;

  /**
   * @brief Construct a new parameters object and parse command line arguments.
   *
   * @param argc Number of command line arguments.
   * @param argv Command line arguments.
   */
  parameters_t(int argc, char** argv) : options(argv[0], "BFS Benchmarking") {
    options.allow_unrecognised_options();
    // Add command line options
    options.add_options()("h,help", "Print help")  // help
        ("m,market", "Matrix file",
         cxxopts::value<std::string>())  // mtx
        ("profile-runs", "Repeated algorithm calls inside one profiling run",
         cxxopts::value<int>()->default_value("1"));

    // Parse command line arguments
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
      help = true;
      std::cout << options.help({""});
      std::cout << "  [optional nvbench args]" << std::endl << std::endl;
      // Do not exit so we also print NVBench help.
    } else {
      if (result.count("market") == 1) {
        filename = result["market"].as<std::string>();
        profile_repetitions = result["profile-runs"].as<int>();
        if (profile_repetitions < 1) {
          std::cerr << "--profile-runs must be positive" << std::endl;
          std::exit(1);
        }
        if (!util::is_market(filename)) {
          std::cout << options.help({""});
          std::cout << "  [optional nvbench args]" << std::endl << std::endl;
          std::exit(0);
        }
      } else {
        std::cout << options.help({""});
        std::cout << "  [optional nvbench args]" << std::endl << std::endl;
        std::exit(0);
      }
    }
  }
};

void bfs_bench(nvbench::state& state) {
  // --
  // Add metrics
  state.collect_dram_throughput();
  state.collect_l1_hit_rates();
  state.collect_l2_hit_rates();
  state.collect_loads_efficiency();
  state.collect_stores_efficiency();

  // --
  // IO
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(filename);

  format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t> csr;
  csr.from_coo(coo);

  // --
  // Build graph

  auto G = graph::build<memory_space_t::device>(properties, csr);

  // --
  // Params and memory allocation
  vertex_t single_source = 0;

  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<vertex_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);

  // --
  // Run BFS with NVBench
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    for (int i = 0; i < profile_repetitions; ++i) {
      gunrock::profiling::nvtx_range_t algorithm_range{"algorithm"};
      gunrock::bfs::run(G, single_source, distances.data().get(),
                        predecessors.data().get());
      algorithm_range.end();
    }
  });
}

int main(int argc, char** argv) {
  parameters_t arguments(argc, argv);
  filename = arguments.filename;

  if (arguments.help) {
    // Print NVBench help.
    const char* args[1] = {"-h"};
    NVBENCH_MAIN_BODY(1, args);
  } else {
    // Remove all gunrock parameters and pass to nvbench.
    auto profile_repetitions_arg = std::to_string(profile_repetitions);
    auto args = filtered_argv(argc, argv, "--market", "-m", "--profile-runs",
                              filename, profile_repetitions_arg);
    NVBENCH_BENCH(bfs_bench);
    NVBENCH_MAIN_BODY(args.size(), args.data());
  }
}
