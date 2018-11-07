#include "PrepareTracks.h"
#include "VeloEventModel.cuh"
#include "ClusteringDefinitions.cuh"
#include "VeloUTDefinitions.cuh"
#include "SciFiDefinitions.cuh"
#include "Tracks.h"
#include "InputTools.h"
#include "TrackChecker.h"
#include "MCParticle.h"
#include "VeloConsolidated.cuh"

template<>
std::vector<trackChecker::Tracks> prepareTracks<TrackCheckerVelo>(
  const uint* track_atomics,
  const uint* track_hit_number_pinned,
  const char* track_hits_pinned,
  const uint number_of_events)
{
  /* Tracks to be checked, save in format for checker */
  std::vector<trackChecker::Tracks> checker_tracks; // all tracks from all events
  for ( uint i_event = 0; i_event < number_of_events; i_event++ ) {
    trackChecker::Tracks tracks; // all tracks within one event
    
    const Velo::Consolidated::Tracks velo_tracks {
      (uint*) track_atomics, (uint*) track_hit_number_pinned, i_event, number_of_events};
    const uint number_of_tracks_event = velo_tracks.number_of_tracks(i_event);

    for ( uint i_track = 0; i_track < number_of_tracks_event; i_track++ ) {
      trackChecker::Track t;
      
      const uint velo_track_number_of_hits = velo_tracks.number_of_hits(i_track);
      Velo::Consolidated::Hits velo_track_hits = velo_tracks.get_hits((uint*) track_hits_pinned, i_track);

      for ( int i_hit = 0; i_hit < velo_track_number_of_hits; ++i_hit ) {
        t.addId(velo_track_hits.LHCbID[i_hit]);
      } 
      tracks.push_back(t);
    } // tracks
    checker_tracks.emplace_back( tracks );
  }
  
  return checker_tracks;
}

template<>
std::vector<trackChecker::Tracks> prepareTracks<TrackCheckerVeloUT, VeloUTTracking::TrackUT> (
  const VeloUTTracking::TrackUT* tracks,
  const int* number_of_tracks,
  const uint number_of_events)
{
  std::vector<trackChecker::Tracks> checker_tracks;
  for (int i_event = 0; i_event < number_of_events; ++i_event) {
    const auto event_number_of_tracks = number_of_tracks[i_event];
    const auto event_tracks_pointer = tracks + i_event * VeloUTTracking::max_num_tracks;

    //debug_cout << "event has " << event_number_of_tracks << " tracks" << std::endl;
    trackChecker::Tracks event_tracks;
    for ( int i_track = 0; i_track < event_number_of_tracks; ++i_track ) {
      const auto& veloUT_track = event_tracks_pointer[i_track];
      trackChecker::Track checker_track;
      assert( veloUT_track.hitsNum < VeloUTTracking::max_track_size);
      //debug_cout << "at track " << std::dec << i_track << std::endl;
      for ( int i_hit = 0; i_hit < veloUT_track.hitsNum; ++i_hit ) {
        //debug_cout<<"\t LHCbIDsVeloUT["<<i_hit<<"] = "<< std::hex << veloUT_track.LHCbIDs[i_hit] << std::endl;
        LHCbID lhcb_id( veloUT_track.LHCbIDs[i_hit] );
        checker_track.addId( lhcb_id );
      }
      event_tracks.push_back(checker_track);
    }
    checker_tracks.emplace_back(event_tracks);
  }
  return checker_tracks;
}

template<>
trackChecker::Tracks prepareTracksSingleEvent<TrackCheckerForward, SciFi::Track> (
  const SciFi::Track* event_tracks_pointer,
  const int event_number_of_tracks)
{
  trackChecker::Tracks event_tracks;
  for ( int i_track = 0; i_track < event_number_of_tracks; i_track++ ) {
    const auto& forward_track = event_tracks_pointer[i_track];
    trackChecker::Track checker_track;
    if ( forward_track.hitsNum >= SciFi::max_track_size )
      debug_cout << "at track " << i_track << " forward track hits Num = " << forward_track.hitsNum << std::endl;
    assert( forward_track.hitsNum < SciFi::max_track_size );
    //debug_cout << "at track " << std::dec << i_track << " with " << forward_track.hitsNum << " hits " << std::endl;
    for ( int i_hit = 0; i_hit < forward_track.hitsNum; ++i_hit ) {
      //debug_cout<<"\t LHCbIDs Forward["<<i_hit<<"] = " << std::hex << forward_track.LHCbIDs[i_hit]<< std::endl;
      LHCbID lhcb_id( forward_track.LHCbIDs[i_hit] );
      checker_track.addId( lhcb_id );
    }
    event_tracks.push_back(checker_track);
  }

  return event_tracks;
}

template<>
std::vector<trackChecker::Tracks> prepareTracks<TrackCheckerForward, SciFi::Track> (
  const SciFi::Track* tracks,
  const int* number_of_tracks,
  const uint number_of_events)
{
  std::vector<trackChecker::Tracks> checker_tracks;
  for ( int i_event = 0; i_event < number_of_events; ++i_event ) {
    const auto event_number_of_tracks = number_of_tracks[i_event];
    const auto event_tracks_pointer = tracks + i_event * SciFi::max_tracks;

    debug_cout << "SciFi checker: Event " << i_event << ": " << event_number_of_tracks << " tracks " << std::endl;

    const auto event_tracks = prepareTracksSingleEvent<TrackCheckerForward, SciFi::Track> (
      event_tracks_pointer,
      event_number_of_tracks
    );

    checker_tracks.emplace_back(event_tracks);
  }
  return checker_tracks;
}
