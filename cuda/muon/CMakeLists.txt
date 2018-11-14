file(GLOB muon_features_extraction "preprocessing/src/*cu")

include_directories(preprocessing/include)
include_directories(../../stream/gear/include)
include_directories(common/include)

cuda_add_library(Muon STATIC
  ${muon_features_extraction}
)
