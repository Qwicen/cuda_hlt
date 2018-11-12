include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)

file(GLOB clustering "src/*cpp")
include_directories(include)
cuda_add_library(CheckClustering STATIC
  ${clustering}
)

target_link_libraries(CheckClustering x86Clustering)
