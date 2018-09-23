file(GLOB scifi_common "common/src/*cu")
file(GLOB scifi_preprocessing "preprocessing/src/*cu")

include_directories(common/include)
include_directories(preprocessing/include)
include_directories(../utils/sorting/include)

cuda_add_library(SciFi STATIC
  ${scifi_common}
  ${scifi_preprocessing}
)
