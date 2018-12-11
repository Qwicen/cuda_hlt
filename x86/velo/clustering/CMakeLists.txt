include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)

file(GLOB x86_clustering "src/*cpp")

# to do: why do I need nvcc to compile the x86 code?
# it depends on clustering functions in cuda/velo/mask_clustering,
# we should make them __host__ and __device__ functions
add_library(x86Clustering SHARED ${x86_clustering})
set_target_properties(x86Clustering PROPERTIES POSITION_INDEPENDENT_CODE ON)
