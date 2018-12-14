file(GLOB muon_features_extraction "preprocessing/src/*cu")
file(GLOB muon_classification "IsMuon/src/*cu")

include_directories(tests)
include_directories(preprocessing/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/gear/include)
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
include_directories(IsMuon/include)

cuda_add_library(Muon STATIC
  ${muon_features_extraction}
  ${muon_classification}
)

macro(test_program name)
cuda_add_executable(${name} tests/${name}.cu)

target_link_libraries(${name} Muon)
endmacro()

test_program(TestFeatures)
