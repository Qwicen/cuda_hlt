file(GLOB muon_features_extraction "preprocessing/src/*cu")
file(GLOB muon_classification "IsMuon/src/*cu")

include_directories(preprocessing/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/gear/include)
include_directories(common/include)
include_directories(IsMuon/include)

cuda_add_library(Muon STATIC
  ${muon_features_extraction}
  ${muon_classification}
)
