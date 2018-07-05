include_directories(include)

cuda_add_library(PatPV SHARED
  src/PVSeedTool.cpp
  src/AdaptivePV3DFitter.cpp
  )

