include_directories(include)
include_directories(SOAContainer/include)

add_library(TrackChecking SHARED
  src/velopix-input-reader.cc
  src/MCAssociator.cc
  src/TrackChecker.cc
  )
