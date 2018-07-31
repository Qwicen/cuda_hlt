include_directories(include)
include_directories(../../main/include)

add_library(TrackChecking SHARED
  src/velopix-input-reader.cc
  src/MCAssociator.cc
  src/TrackChecker.cc
  )
