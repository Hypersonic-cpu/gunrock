include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Cloning External Project: NVBench")
get_filename_component(FC_BASE "${PROJECT_SOURCE_DIR}/externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

FetchContent_Declare(
    nvbench
    GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
    # Keep the benchmark driver API and CMake requirements stable.  NVBench
    # main is a moving target and newer revisions currently require CMake 4.
    GIT_TAG        182c77e4f42d86096505fbb0d3795eb7f013439c
)

FetchContent_GetProperties(nvbench)
if(NOT nvbench_POPULATED)
  FetchContent_MakeAvailable(
    nvbench
  )
endif()

# Exposing nvbench's source and include directory
set(NVBENCH_INCLUDE_DIR "${nvbench_SOURCE_DIR}")
set(NVBENCH_BUILD_DIR "${nvbench_BINARY_DIR}")

# Add subdirectory ::nvbench
# add_subdirectory(${NVBENCH_INCLUDE_DIR} ${NVBENCH_BUILD_DIR})
