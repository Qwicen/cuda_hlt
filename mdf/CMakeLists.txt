set(SOURCES
  src/compression.cpp
  src/raw_helpers.cpp
  src/read_mdf.cpp)

add_library(mdf ${SOURCES})
target_include_directories (mdf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_include_directories(
  mdf PUBLIC
  ${CMAKE_SOURCE_DIR}/main/include
  ${ZLIB_INCLUDE_DIRS})

target_link_libraries(
  mdf PUBLIC
  ${ZLIB_LIBRARIES})

if (LZMA_FOUND)
  target_include_directories(
	mdf PUBLIC
	${LIBLZMA_INCLUDE_DIRS})

  target_link_libraries(
	mdf PUBLIC
	${LIBLZMA_LIBRARIES})

  target_compile_definitions(
	mdf	PRIVATE
	"-DHAVE_LZMA")
endif (LZMA_FOUND)

if (LZ4_FOUND)
  target_include_directories(
	mdf PUBLIC
	${LIBLZ4_INCLUDE_DIRS})

  target_link_libraries(
	mdf
	${LIBLZ4_LIBRARIES})

  target_compile_definitions(
	mdf	PRIVATE
	"-DHAVE_LZ4")
endif (LZ4_FOUND)

macro(test_program name)
add_executable(${name}
  test/${name}.cpp)

target_include_directories(${name} PUBLIC
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${CMAKE_SOURCE_DIR}/mdf/include
  ${CMAKE_SOURCE_DIR}/main/include
  ${CMAKE_SOURCE_DIR}/cuda/event_model/velo/include
  ${CMAKE_SOURCE_DIR}/cuda/event_model/common/include
  ${CMAKE_SOURCE_DIR}/cuda/velo/common/include
  ${CMAKE_SOURCE_DIR}/cuda/SciFi/common/include
  ${CMAKE_SOURCE_DIR}/cuda/muon/common/include
  ${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/include
  ${CMAKE_SOURCE_DIR}/cuda/UT/common/include
  ${CMAKE_SOURCE_DIR}/cuda/UT/PrVeloUT/include
  ${CMAKE_SOURCE_DIR}/checker/tracking/include)

target_link_libraries(${name} Common Velo)
endmacro()

test_program(test_read)
test_program(test_read_bin)
test_program(dump_banks)
