include_directories(include)
include_directories(../cuda/velo/common/include)
include_directories(../cuda/patPV/include)
include_directories(../cuda/event_model/velo/include)

add_library(x86PatPV SHARED
  src/PVSeedTool.cpp
  src/AdaptivePV3DFitter.cpp
  )
