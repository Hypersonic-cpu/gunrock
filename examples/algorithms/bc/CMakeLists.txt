add_essentials_example(bc)

find_package(OpenMP REQUIRED COMPONENTS CXX)
target_link_libraries(bc PRIVATE OpenMP::OpenMP_CXX)
if(ESSENTIALS_NVIDIA_BACKEND)
    target_compile_options(bc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
endif()
