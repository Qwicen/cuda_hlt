include_directories(include)

add_library(TrackChecking SHARED
  src/velopix-input-reader.cc
  src/MCAssociator.cc
  src/TrackChecker.cc
)
