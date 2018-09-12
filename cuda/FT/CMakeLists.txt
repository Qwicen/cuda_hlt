file(GLOB ft_common "common/src/*cu")
file(GLOB ft_preprocessing "preprocessing/src/*cu")

include_directories(common/include)
include_directories(preprocessing/include)

cuda_add_library(FT STATIC
  ${ft_common}
  ${ft_preprocessing}
)
