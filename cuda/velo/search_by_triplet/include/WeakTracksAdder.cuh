#pragma once

#include "../../common/include/VeloDefinitions.cuh"

__device__ void weakTracksAdder(
  int* shared_hits,
  unsigned int* weaktracks_insertPointer,
  unsigned int* tracks_insertPointer,
  unsigned int* weak_tracks,
  TrackHits* tracklets,
  TrackHits* tracks,
  bool* hit_used
);

__device__ void weakTracksAdderShared(
  int* shared_hits,
  unsigned int* weaktracks_insertPointer,
  unsigned int* tracks_insertPointer,
  unsigned int* weak_tracks,
  TrackHits* tracklets,
  TrackHits* tracks,
  bool* hit_used
);
