#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iterator>

struct filtered_argv_t {
  std::vector<std::string> storage;
  std::vector<char*> pointers;

  std::size_t size() const { return pointers.size(); }
  char** data() { return pointers.data(); }
};

/**
 * @brief Filter the arguments by removing the strings.
 *
 * @tparam t string typename.
 * @param argc Argument count.
 * @param argv Arguments.
 * @param s Variadic parameters.
 * @return auto A vector of char* with s removed.
 */
template <typename... t>
auto filtered_argv(int argc, char** argv, t&... s) {
  filtered_argv_t result;
  result.storage.assign(argv, argv + argc);
  auto condition = [s...](const std::string& arg) -> bool {
    return ((arg == s) || ...);
  };
  result.storage.erase(
      std::remove_if(result.storage.begin(), result.storage.end(), condition),
      result.storage.end());

  result.pointers.reserve(result.storage.size());

  for (auto& arg : result.storage)
    result.pointers.push_back(arg.data());

  return result;
}

void print_arg(int argc, char** argv) {
  std::cout << "name of program: " << argv[0] << '\n';
  if (argc > 1) {
    std::cout << "there are " << argc - 1 << " (more) arguments, they are:\n";
    std::copy(argv + 1, argv + argc,
              std::ostream_iterator<const char*>(std::cout, "\n"));
  }
}
