find_path(flatbuffers_INCLUDE_DIR NAMES flatbuffers/flatbuffers.h
          HINTS ${flatbuffers_home}/include/)

find_program(flatbuffers_FLATC_EXECUTABLE NAMES flatc
          HINTS ${flatbuffers_home})

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(flatbuffers 
  DEFAULT_MSG flatbuffers_INCLUDE_DIR flatbuffers_FLATC_EXECUTABLE)

mark_as_advanced(flatbuffers_FOUND flatbuffers_INCLUDE_DIR)
mark_as_advanced(flatbuffers_EXEC_FOUND flatbuffers_FLATC_EXECUTABLE)
set(flatbuffers_INCLUDE_DIRS ${flatbuffers_INCLUDE_DIR})
set(flatbuffers_FLATC ${flatbuffers_FLATC_EXECUTABLE})