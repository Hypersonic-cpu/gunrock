#include <vector>
#include <string>

#include <gunrock/algorithms/tc.hxx>
#include <gunrock/io/parameters.hxx>
#include "tc_cpu.hxx"

#include <cxxopts.hpp>

using namespace gunrock;
using namespace memory;

struct parameters_t {
  std::string filename;
  cxxopts::Options options;
  bool binary = false;
  std::string binary_output_filename;
  bool validate;
  bool reduce_all_triangles;
  std::string advance_load_balance;

  /**
   * @brief Construct a new parameters object and parse command line arguments.
   *
   * @param argc Number of command line arguments.
   * @param argv Command line arguments.
   */
  parameters_t(int argc, char** argv)
      : options(argv[0], "Traingle Counting example") {
    // Add command line options
    options.add_options()("help", "Print help")(
        "validate", "CPU validation",
        cxxopts::value<bool>()->default_value("false"))(
        "m,market", "MatrixMarket file", cxxopts::value<std::string>())(
        "binary-in", "Read graph from a binary CSR file",
        cxxopts::value<std::string>())(
        "binary-out", "Write CSR to a binary file after MatrixMarket input",
        cxxopts::value<std::string>())(
        "r,reduce",
        "Compute a single triangle count for the entire graph (default = "
        "false)",
        cxxopts::value<bool>()->default_value("false"))(
        "advance_load_balance",
        "Load balancing technique for the advance operator",
        cxxopts::value<std::string>()->default_value("block_mapped"));

    // Parse command line arguments
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }

    const bool has_market = result.count("market") != 0;
    const bool has_binary_in = result.count("binary-in") != 0;
    const bool has_binary_out = result.count("binary-out") != 0;
    auto fail = [](const std::string& message) {
      std::cerr << "Error: " << message << std::endl;
      std::exit(EXIT_FAILURE);
    };

    if (has_market && has_binary_in) {
      fail("--market and --binary-in are mutually exclusive");
    }
    if (has_binary_out && !has_market) {
      fail("--binary-out requires --market MatrixMarket input");
    }
    if (!has_market && !has_binary_in) {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }

    if (has_market) {
      filename = result["market"].as<std::string>();
      binary = util::is_binary_csr(filename);
      if (!binary && !util::is_market(filename)) {
        fail("--market must name a MatrixMarket file");
      }
      if (has_binary_out && binary) {
        fail("--binary-out requires MatrixMarket input, not binary CSR input");
      }
    } else {
      filename = result["binary-in"].as<std::string>();
      binary = true;
    }
    if (has_binary_out) {
      binary_output_filename = result["binary-out"].as<std::string>();
    }

    validate = result["validate"].as<bool>();
    reduce_all_triangles = result["reduce"].as<bool>();
    advance_load_balance = result["advance_load_balance"].as<std::string>();
  }
};

void test_tc(int num_arguments, char** argument_array) {
  // --
  // Define types

  using vertex_t = uint32_t;
  using edge_t = uint32_t;
  using weight_t = float;
  using count_t = vertex_t;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  csr_t csr;

  // --
  // IO
  parameters_t arguments(num_arguments, argument_array);
  gunrock::graph::graph_properties_t properties =
      gunrock::graph::graph_properties_t();

  if (arguments.binary) {
    csr.read_binary(arguments.filename);
  } else {
    io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
    auto [market_properties, coo] = mm.load(arguments.filename);
    properties = market_properties;
    if (!properties.symmetric) {
      std::cerr << "Error: input matrix must be symmetric" << std::endl;
      exit(1);
    }
    csr.from_coo(coo);
    if (!arguments.binary_output_filename.empty()) {
      csr.write_binary(arguments.binary_output_filename);
    }
  }

  // --
  // Build graph

  auto G = graph::build<memory_space_t::device>(properties, csr);
  std::cout << "Graph vertices : " << G.get_number_of_vertices() << std::endl;
  std::cout << "Graph CSR edges : " << G.get_number_of_edges() << std::endl;

  // --
  // Params and memory allocation

  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<count_t> triangles_count(n_vertices, 0);

  // --
  // GPU Run

  // Create context
  auto context = std::make_shared<gcuda::multi_context_t>(0);

  // Create param and result structs
  gunrock::options_t options(
      gunrock::io::cli::parse_load_balance(arguments.advance_load_balance));
  tc::param_t<vertex_t> param(arguments.reduce_all_triangles, options);
  std::size_t total_triangles = 0;
  tc::result_t<vertex_t> result(triangles_count.data().get(), &total_triangles);

  float gpu_elapsed = tc::run(G, param, result, context);

  // --
  // Log

  print::head(triangles_count, 40, "Per-vertex triangle count");
  if (arguments.reduce_all_triangles) {
    std::cout << "Total Graph Traingles : " << total_triangles << std::endl;
  }
  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;

  // --
  // CPU validation
  if (arguments.validate) {
    std::vector<count_t> reference_triangles_count(n_vertices, 0);
    std::size_t reference_total_triangles = 0;

    float cpu_elapsed =
        tc_cpu::run(csr, reference_triangles_count, reference_total_triangles);
    uint32_t n_errors = 0;
    if (total_triangles != reference_total_triangles) {
      std::cout << "Error: Total TC mismatch: " << total_triangles
                << "! = " << reference_total_triangles << std::endl;
      n_errors++;
    }
    n_errors += util::compare(
        triangles_count.data().get(), reference_triangles_count.data(),
        n_vertices, [](const auto x, const auto y) { return x != y; }, true);
    std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
    std::cout << "Number of errors : " << n_errors << std::endl;
  }
}

int main(int argc, char** argv) {
  test_tc(argc, argv);
}
