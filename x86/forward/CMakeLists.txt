include_directories(include)
include_directories(../../checker/tracking/include)
include_directories(../../main/include)
include_directories(../../PrForward/include)
include_directories(../../cuda/veloUT/common/include)
include_directories(../../cuda/veloUT/PrVeloUT/include)
include_directories(../../cuda/forward/common/include)
include_directories(../../cuda/velo/common/include)

file(GLOB x86Forward_cpp "src/*cpp" "../../PrForward/src/*cpp")

cuda_add_library(x86Forward STATIC
  ${x86Forward_cpp}
)
if ( ROOT_FOUND )
  target_compile_definitions(x86Forward PUBLIC WITH_ROOT)
  target_link_libraries(x86Forward
    ${ROOT_LIBRARIES}
  )
endif()
