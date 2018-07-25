include_directories(include)

cuda_add_library(Forward SHARED
  src/PrForward.h
  src/PrForward.cpp
  )

