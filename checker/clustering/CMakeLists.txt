include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)

file(GLOB clustering "src/*cpp")
include_directories(include)
add_library(CheckClustering STATIC
  ${clustering}
)

target_link_libraries(CheckClustering PRIVATE x86Clustering)
set_target_properties(CheckClustering PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_property(TARGET CheckClustering PROPERTY
             CUDA_SEPARABLE_COMPILATION ON)
