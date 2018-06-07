include_directories(lib/include)
include_directories(lib/SOAContainer/include)

add_subdirectory(lib)

file(GLOB clustering "clustering/src/*cpp")
include_directories(clustering/include)
cuda_add_library(CheckClustering STATIC
  ${clustering}
)

target_link_libraries(CheckClustering x86Clustering)
