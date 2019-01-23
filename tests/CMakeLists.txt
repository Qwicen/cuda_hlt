include_directories(${CMAKE_SOURCE_DIR}/stream/gear/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/muon/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/muon/preprocessing/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/muon/IsMuon/include)

set(UNIT_TEST_LIST
  MuonFeaturesExtraction)
 
foreach(NAME IN LISTS UNIT_TEST_LIST)

  set(TARGET_NAME ${NAME}.test)
  
  add_executable(${TARGET_NAME}
    main.cu
    ${NAME}.test.cu)
  
  target_link_libraries(${TARGET_NAME}
    PUBLIC Muon)

  set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  
  add_test(
    NAME ${TARGET_NAME}
    COMMAND ${TARGET_NAME})
endforeach()
