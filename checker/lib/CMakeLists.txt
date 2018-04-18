include_directories(include)
include_directories(SOAContainer/include)


add_library( libTrackChecking SHARED
  src/velopix-input-reader.cc
  src/TrackSerializer.cc
  src/MCAssociator.cc
  src/TrackChecker.cc
  )
