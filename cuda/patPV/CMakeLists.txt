file(GLOB patPV "src/patPV.cu")
include_directories(include)
include_directories(../../main/include)
include_directories(../velo/common/include)
include_directories(../veloUT/common/include)

cuda_add_library(PatPV STATIC
  ${patPV} 
)
