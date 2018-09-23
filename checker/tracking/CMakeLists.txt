include_directories(include)
include_directories(../../main/include)
include_directories(../../cuda/UT/PrVeloUT/include)

add_library(TrackChecking SHARED
  src/velopix-input-reader.cc
  src/MCAssociator.cc
  src/TrackChecker.cc
)
