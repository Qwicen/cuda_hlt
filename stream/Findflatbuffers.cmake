find_path(flatbuffers_INCLUDE_DIR NAMES flatbuffers/flatbuffers.h
          HINTS ${flatbuffers_home}/include/)

find_program(flatbuffers_FLATC_EXECUTABLE NAMES flatc
          HINTS ${flatbuffers_home})

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(flatbuffers 
  DEFAULT_MSG flatbuffers_INCLUDE_DIR flatbuffers_FLATC_EXECUTABLE)

if(flatbuffers_FOUND)
  set(flatbuffers_INCLUDE_DIRS ${flatbuffers_INCLUDE_DIR})
  set(flatbuffers_FLATC ${flatbuffers_FLATC_EXECUTABLE})
endif()