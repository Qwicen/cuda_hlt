include_directories(include)
include_directories(SOAContainer/include)

find_package( ROOT REQUIRED COMPONENTS  Matrix Physics MathCore Thread GenVector ) 

add_library( libTrackChecking SHARED
  src/velopix-input-reader.cc
  src/MCAssociator.cc
  src/TrackChecker.cc
  )

target_link_libraries( libTrackChecking ${ROOT_LIBRARIES} )
