#pragma once

#include <gunrock/cuda/cuda.hxx>
#include <gunrock/framework/operators/configs.hxx>

namespace gunrock::operators::advance::swpf {

__device__ __forceinline__ void prefetch_l1(const void* ptr) {
#if __HIP_PLATFORM_NVIDIA__
  asm volatile("prefetch.global.L1 [%0];" : : "l"(ptr));
#endif
}

__device__ __forceinline__ void prefetch_l2(const void* ptr) {
#if __HIP_PLATFORM_NVIDIA__
  asm volatile("prefetch.global.L2 [%0];" : : "l"(ptr));
#endif
}

template <swpf_target_t target>
__device__ __forceinline__ void prefetch_read(const void* ptr) {
  if constexpr (target == swpf_target_t::l1)
    prefetch_l1(ptr);
  else
    prefetch_l2(ptr);
}

// Global atomic operations always access L2, independent of the read target.
__device__ __forceinline__ void prefetch_atomic_target(const void* ptr) {
  prefetch_l2(ptr);
}

}  // namespace gunrock::operators::advance::swpf
