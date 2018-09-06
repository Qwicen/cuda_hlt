include_directories(include)
include_directories(../../../cuda/veloUT/PrVeloUT/include)
include_directories(../../../cuda/velo/common/include)
include_directories(../../../cuda/veloUT/common/include)
include_directories(../../../cuda/utils/include)
include_directories(../../../main/include)
include_directories(../../../checker/tracking/include)

file(GLOB x86VeloUT_cpp "src/*cpp")

cuda_add_library(x86VeloUT STATIC
  ${x86VeloUT_cpp}
)
if ( ROOT_FOUND )
  target_compile_definitions(x86VeloUT PUBLIC WITH_ROOT)
  message("Found Root, compiling WITH_ROOT")
  target_link_libraries(x86VeloUT
    ${ROOT_LIBRARIES}
  )
endif()
