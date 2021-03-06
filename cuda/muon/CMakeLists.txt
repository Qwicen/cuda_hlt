file(GLOB muon_features_extraction "preprocessing/src/*cu")
file(GLOB muon_classification "classification/src/*cu")
file(GLOB muon_filtering "is_muon/src/*cu")

include_directories(preprocessing/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/gear/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/setup/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/consolidate/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/velo/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/SciFi/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/common/include)
include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(common/include)
include_directories(is_muon/include)
include_directories(classification/include)

add_library(Muon STATIC
  ${muon_filtering}
  ${muon_features_extraction}
  ${muon_classification}
)
set_property(TARGET Muon PROPERTY CUDA_SEPARABLE_COMPILATION ON)
