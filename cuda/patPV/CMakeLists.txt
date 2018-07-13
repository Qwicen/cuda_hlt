file(GLOB patPV "src/*cu")
include_directories(include)

cuda_add_library(PatPV STATIC
  ${patPV}
)
