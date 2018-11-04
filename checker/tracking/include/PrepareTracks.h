#pragma once

#include <vector>
#include "TrackChecker.h"
#include "Logger.h"

/**
 * @brief Prepares tracks for non-consolidated datatypes.
 */
template<typename T, typename R>
std::vector<trackChecker::Tracks> prepareTracks(
  const R* tracks,
  const int* number_of_tracks,
  const uint number_of_events);

/**
 * @brief Prepares tracks for a single event.
 */
template<typename T, typename R>
trackChecker::Tracks prepareTracksSingleEvent(
  const R* event_tracks_pointer,
  const int event_number_of_tracks);

/**
 * @brief Prepares tracks for consolidated datatypes.
 */
template<typename T>
std::vector<trackChecker::Tracks> prepareTracks(
  const uint* track_atomics,
  const uint* track_hit_number_pinned,
  const char* track_hits_pinned,
  const uint number_of_events);
