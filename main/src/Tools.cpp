#include "Tools.h"

bool check_velopix_events(
  const std::vector<char>& events,
  const std::vector<uint>& event_offsets,
  int n_events
) {
  int error_count = 0;
  int n_sps_all_events = 0;
  for ( int i_event = 0; i_event < n_events; ++i_event ) {
    const char* raw_input = events.data() + event_offsets[i_event];

    const char* p = events.data() + event_offsets[i_event];
    uint32_t number_of_raw_banks = *((uint32_t*)p); p += sizeof(uint32_t);
    uint32_t* raw_bank_offset = (uint32_t*) p; p += number_of_raw_banks * sizeof(uint32_t);

    uint32_t sensor =  *((uint32_t*)p);  p += sizeof(uint32_t);
    uint32_t sp_count =  *((uint32_t*)p); p += sizeof(uint32_t);

    const auto raw_event = VeloRawEvent(raw_input);
    int n_sps_event = 0;
    for ( int i_raw_bank = 0; i_raw_bank < raw_event.number_of_raw_banks; i_raw_bank++ ) {
      const auto raw_bank = VeloRawBank(raw_event.payload + raw_event.raw_bank_offset[i_raw_bank]);
      n_sps_event += raw_bank.sp_count;
      if ( i_raw_bank != raw_bank.sensor_index ) {
        error_cout << "at raw bank " << i_raw_bank << ", but index = " << raw_bank.sensor_index << std::endl;
        ++error_count;
      }
      if ( raw_bank.sp_count > 0 ) {
        uint32_t sp_word = raw_bank.sp_word[0];
        uint8_t sp = sp_word & 0xFFU;
        if (0 == sp) { continue; };
        const uint32_t sp_addr = (sp_word & 0x007FFF00U) >> 8;
        const uint32_t sp_row = sp_addr & 0x3FU;
        const uint32_t sp_col = (sp_addr >> 6);
        const uint32_t no_sp_neighbours = sp_word & 0x80000000U;
      }
    }
    n_sps_all_events += n_sps_event;
  }

  if (error_count>0) {
    error_cout << error_count << " errors detected." << std::endl;
    return false;
  }
  return true;
}

/**
 * @brief Obtains results statistics.
 */
std::map<std::string, float> calcResults(std::vector<float>& times){
    // sqrt ( E( (X - m)2) )
    std::map<std::string, float> results;
    float deviation = 0.0f, variance = 0.0f, mean = 0.0f, min = FLT_MAX, max = 0.0f;

    for(auto it = times.begin(); it != times.end(); it++){
        const float seconds = (*it);
        mean += seconds;
        variance += seconds * seconds;

        if (seconds < min) min = seconds;
        if (seconds > max) max = seconds;
    }

    mean /= times.size();
    variance = (variance / times.size()) - (mean * mean);
    deviation = std::sqrt(variance);

    results["variance"] = variance;
    results["deviation"] = deviation;
    results["mean"] = mean;
    results["min"] = min;
    results["max"] = max;

    return results;
}

std::vector<trackChecker::Tracks> prepareTracks(
  uint* host_velo_tracks_atomics,
  uint* host_velo_track_hit_number_pinned,
  char* host_velo_track_hits_pinned,
  const uint number_of_events
) {
  /* Tracks to be checked, save in format for checker */
  std::vector< trackChecker::Tracks > all_tracks; // all tracks from all events
  for ( uint i_event = 0; i_event < number_of_events; i_event++ ) {
    trackChecker::Tracks tracks; // all tracks within one event
    
    const Velo::Consolidated::Tracks velo_tracks {host_velo_tracks_atomics, host_velo_track_hit_number_pinned, i_event, number_of_events};
    const uint number_of_tracks_event = velo_tracks.number_of_tracks(i_event);

    for ( uint i_track = 0; i_track < number_of_tracks_event; i_track++ ) {
      trackChecker::Track t;
      
      const uint velo_track_number_of_hits = velo_tracks.number_of_hits(i_track);
      Velo::Consolidated::Hits velo_track_hits = velo_tracks.get_hits((uint*) host_velo_track_hits_pinned, i_track);

      for ( int i_hit = 0; i_hit < velo_track_number_of_hits; ++i_hit ) {
        t.addId(velo_track_hits.LHCbID[i_hit]);
      } 
      tracks.push_back( t );
    } // tracks
    all_tracks.emplace_back( tracks );
  }
  
  return all_tracks;
}

trackChecker::Tracks prepareVeloUTTracksEvent(
  const VeloUTTracking::TrackUT* veloUT_tracks,
  const int n_veloUT_tracks
) {
  //debug_cout << "event has " << n_veloUT_tracks << " tracks" << std::endl;
  trackChecker::Tracks checker_tracks;
  for ( int i_track = 0; i_track < n_veloUT_tracks; ++i_track ) {
    VeloUTTracking::TrackUT veloUT_track = veloUT_tracks[i_track];
    trackChecker::Track checker_track;
    assert( veloUT_track.hitsNum < VeloUTTracking::max_track_size);
    //debug_cout << "at track " << std::dec << i_track << std::endl;
    for ( int i_hit = 0; i_hit < veloUT_track.hitsNum; ++i_hit ) {
      //debug_cout<<"\t LHCbIDsVeloUT["<<i_hit<<"] = "<< std::hex << veloUT_track.LHCbIDs[i_hit] << std::endl;
      LHCbID lhcb_id( veloUT_track.LHCbIDs[i_hit] );
      checker_track.addId( lhcb_id );
    }
    checker_tracks.push_back( checker_track );
  }

  return checker_tracks;
}

trackChecker::Tracks prepareForwardTracksVeloUTOnly(
  std::vector< VeloUTTracking::TrackUT > forward_tracks
) {
  trackChecker::Tracks checker_tracks;
  int i_track = 0;
  for ( VeloUTTracking::TrackUT forward_track : forward_tracks ) {
    trackChecker::Track checker_track;
    //debug_cout << "at track " << std::dec << i_track << std::endl;
    for ( int i_hit = 0; i_hit < forward_track.hitsNum; ++i_hit ) {
      // debug_cout<<"\t LHCbIDsForward["<<i_hit<<"] = " << std::hex << forward_track.LHCbIDs[i_hit]<< std::endl;
      LHCbID lhcb_id( forward_track.LHCbIDs[i_hit] );
      checker_track.addId( lhcb_id );
    }
    checker_tracks.push_back( checker_track );
    ++i_track;
  }
  //debug_cout<<"end prepareForwardTracks"<<std::endl;

  return checker_tracks;
}

std::vector< trackChecker::Tracks > prepareForwardTracks(
  SciFi::Track* scifi_tracks,
  uint* n_scifi_tracks,
  const int number_of_events
) {
  std::vector< trackChecker::Tracks > checker_tracks;
  for ( int i_event = 0; i_event < number_of_events; ++i_event ) {
    debug_cout << "in event " << i_event << " found " << n_scifi_tracks[i_event] << " tracks " << std::endl;
    trackChecker::Tracks ch_tracks = prepareForwardTracksEvent(
      scifi_tracks + i_event * SciFi::max_tracks,
      n_scifi_tracks[i_event]);
    checker_tracks.push_back( ch_tracks );
  }
  return checker_tracks;
}

trackChecker::Tracks prepareForwardTracksEvent(
  SciFi::Track forward_tracks[SciFi::max_tracks],
  const uint n_forward_tracks
) {
  trackChecker::Tracks checker_tracks;
  for ( int i_track = 0; i_track < n_forward_tracks; i_track++ ) {
    const SciFi::Track& forward_track = forward_tracks[i_track];
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
    checker_tracks.push_back( checker_track );
  }
  
  return checker_tracks;
} 

std::vector< trackChecker::Tracks > prepareVeloUTTracks(
  const VeloUTTracking::TrackUT* veloUT_tracks,
  const int* n_veloUT_tracks,
  const int number_of_events
) {
  std::vector< trackChecker::Tracks > checker_tracks;
  for ( int i_event = 0; i_event < number_of_events; ++i_event ) {
    trackChecker::Tracks ch_tracks = prepareVeloUTTracksEvent(
      veloUT_tracks + i_event * VeloUTTracking::max_num_tracks,
      n_veloUT_tracks[i_event]);
    checker_tracks.push_back( ch_tracks );
  }
  return checker_tracks;
}

void call_pr_checker(
  const std::vector< trackChecker::Tracks >& all_tracks,
  const std::string& folder_name_MC,
  const uint start_event_offset,
  const std::string& trackType
) {
  if ( trackType == "Velo" ) {
    call_pr_checker_impl<TrackCheckerVelo> (
      all_tracks,
      folder_name_MC,
      start_event_offset,
      trackType);
  }
  else if ( trackType == "VeloUT" ) {
    call_pr_checker_impl<TrackCheckerVeloUT> (
      all_tracks,
      folder_name_MC,
      start_event_offset,
      trackType);
  }
  else if ( trackType == "Forward" ) {
    call_pr_checker_impl<TrackCheckerForward> (
      all_tracks,
      folder_name_MC,
      start_event_offset,
      trackType);
  }
  else {
    error_cout << "Unknown track type: " << trackType << std::endl;
  }
}

