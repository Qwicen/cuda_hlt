file(GLOB ft_preprocessing "preprocessing/src/*cu")
include_directories(preprocessing/include)

cuda_add_library(Ft STATIC
        ${ft_preprocessing}
        )
