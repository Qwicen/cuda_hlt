include_directories(lib/include)
include_directories(lib/SOAContainer/include)

#add_subdirectory(SOAContainer)
#add_subdirectory(src)

add_library(libvelophidrdz2 SHARED
  src/velopix-input-reader.cc
  src/TrackSerializer.cc
  src/MCAssociator.cc
  src/TrackChecker.cc
  ) 

add_executable(velopix-input-reader velopix-input-reader.cc)
add_executable(checkEfficiencies checkEfficiencies.cc)
target_link_libraries(velopix-input-reader libvelophidrdz2)
target_link_libraries(checkEfficiencies libvelophidrdz2)

