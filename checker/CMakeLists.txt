include_directories(lib/include)
include_directories(lib/SOAContainer/include)

add_subdirectory(lib)

add_executable(velopix-input-reader velopix-input-reader.cc)
add_executable(checkEfficiencies checkEfficiencies.cc)

target_link_libraries(velopix-input-reader libTrackChecking)
target_link_libraries(checkEfficiencies libTrackChecking)

file(GLOB clustering "clustering/src/*cpp")
include_directories(clustering/include)
cuda_add_library(libCheckClustering STATIC
  ${clustering}
)
target_link_libraries(libCheckClustering libx86Clustering)
