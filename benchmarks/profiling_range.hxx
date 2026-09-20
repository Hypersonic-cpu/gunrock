#pragma once

#include <cstdlib>
#include <stdexcept>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

namespace gunrock::profiling {

class nvtx_range_t {
 public:
  explicit nvtx_range_t(const char* name) : active_(is_enabled()) {
    if (!active_) return;

    synchronize();
    nvtxRangePushA(name);
  }

  nvtx_range_t(const nvtx_range_t&) = delete;
  nvtx_range_t& operator=(const nvtx_range_t&) = delete;

  ~nvtx_range_t() {
    if (active_) nvtxRangePop();
  }

  void end() {
    if (!active_) return;

    const auto status = cudaDeviceSynchronize();
    nvtxRangePop();
    active_ = false;
    if (status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(status));
    }
  }

 private:
  static bool is_enabled() {
    const auto* value = std::getenv("GUNROCK_PROFILE_NVTX");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
  }

  static void synchronize() {
    const auto status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(status));
    }
  }

  bool active_;
};

}  // namespace gunrock::profiling
