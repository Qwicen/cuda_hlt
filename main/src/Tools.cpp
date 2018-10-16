#include "Tools.h"

bool check_velopix_events(
  const std::vector<char> events,
  const std::vector<uint> event_offsets,
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
void read_scifi_events_into_arrays( SciFi::HitsSoA *hits_layers_events,
                                 uint32_t n_hits_layers_events[][SciFi::Constants::n_zones],
                                 const std::vector<char> events,
                                 const std::vector<unsigned int> event_offsets,
                                 const int n_events ) {


  for ( int i_event = 0; i_event < n_events; ++i_event ) {
    const char* raw_input = events.data() + event_offsets[i_event];
    int n_hits_total = 0;
    int accumulated_hits = 0;
    int accumulated_hits_layers[12];
    for ( int i_layer = 0; i_layer < SciFi::Constants::n_zones; ++i_layer ) {
      n_hits_layers_events[i_event][i_layer] = *((uint32_t*)raw_input);
      n_hits_total += n_hits_layers_events[i_event][i_layer];
      raw_input += sizeof(uint32_t);
      assert( n_hits_total < SciFi::Constants::max_numhits_per_event );
      hits_layers_events[i_event].layer_offset[i_layer] = accumulated_hits;
      accumulated_hits += n_hits_layers_events[i_event][i_layer];
    }

    for ( int i_layer = 0; i_layer < SciFi::Constants::n_zones; ++i_layer ) {
      int layer_offset = hits_layers_events[i_event].layer_offset[i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &( hits_layers_events[i_event].m_x[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].m_z[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].m_w[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].m_dxdy[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].m_dzdy[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].m_yMin[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].m_yMax[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((unsigned int*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].m_LHCbID[ layer_offset ]) );
      raw_input += sizeof(unsigned int) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((int*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].m_planeCode[ layer_offset ]) );
      raw_input += sizeof(int) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((int*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].m_hitZone[ layer_offset ]) );
      raw_input += sizeof(int) * n_hits_layers_events[i_event][i_layer];

      for ( int i_hit = 0; i_hit < n_hits_layers_events[i_event][i_layer]; ++i_hit ) {
        hits_layers_events[i_event].m_planeCode[ layer_offset + i_hit ] = i_layer;
      }
    }
  }
}

void read_muon_events_into_arrays( Muon::HitsSoA *muon_station_hits,
                                 const std::vector<char> events,
                                 const std::vector<unsigned int> event_offsets,
                                 const int n_events ) {
  
  for ( int i_event = 0; i_event < n_events; ++i_event ) {

    const char* raw_input = events.data() + event_offsets[i_event];      
    std::copy_n((int*) raw_input, Muon::Constants::n_stations, muon_station_hits[i_event].m_number_of_hits_per_station);
    raw_input += sizeof(int) * Muon::Constants::n_stations;

    muon_station_hits[i_event].m_station_offsets[0] = 0;
    for(int i_station = 1; i_station < Muon::Constants::n_stations; ++i_station) {
      muon_station_hits[i_event].m_station_offsets[i_station] = muon_station_hits[i_event].m_station_offsets[i_station - 1] + muon_station_hits[i_event].m_number_of_hits_per_station[i_event - 1];
    }
    
    for(int i_station = 0; i_station < Muon::Constants::n_stations; ++i_station) {
      const int station_offset = muon_station_hits[i_event].m_station_offsets[i_station];
      const int number_of_hits = muon_station_hits[i_event].m_number_of_hits_per_station[i_station];

      std::copy_n((int*) raw_input, number_of_hits, &( muon_station_hits[i_event].m_tile[station_offset]) );
      raw_input += sizeof(int) * number_of_hits;

      std::copy_n((float*) raw_input, number_of_hits, &( muon_station_hits[i_event].m_x[station_offset]) );
      raw_input += sizeof(int) * number_of_hits;

      std::copy_n((float*) raw_input, number_of_hits, &( muon_station_hits[i_event].m_dx[station_offset]) );
      raw_input += sizeof(int) * number_of_hits;

      std::copy_n((float*) raw_input, number_of_hits, &( muon_station_hits[i_event].m_y[station_offset]) );
      raw_input += sizeof(int) * number_of_hits;

      std::copy_n((float*) raw_input, number_of_hits, &( muon_station_hits[i_event].m_dy[station_offset]) );
      raw_input += sizeof(int) * number_of_hits;

      std::copy_n((float*) raw_input, number_of_hits, &( muon_station_hits[i_event].m_z[station_offset]) );
      raw_input += sizeof(int) * number_of_hits;

      std::copy_n((float*) raw_input, number_of_hits, &( muon_station_hits[i_event].m_dz[station_offset]) );
      raw_input += sizeof(int) * number_of_hits;

      std::copy_n((int*) raw_input, number_of_hits, &( muon_station_hits[i_event].m_uncrossed[station_offset]) );
      raw_input += sizeof(int) * number_of_hits;

      std::copy_n((unsigned int*) raw_input, number_of_hits, &( muon_station_hits[i_event].m_time[station_offset]) );
      raw_input += sizeof(int) * number_of_hits;

      std::copy_n((int*) raw_input, number_of_hits, &( muon_station_hits[i_event].m_delta_time[station_offset]) );
      raw_input += sizeof(int) * number_of_hits;

      std::copy_n((int*) raw_input, number_of_hits, &( muon_station_hits[i_event].m_cluster_size[station_offset]) );
      raw_input += sizeof(int) * number_of_hits;
    }
  }
}


// void check_ut_events(
//   const VeloUTTracking::HitsSoA *hits_layers_events,
//   const int n_events
// ) {
//   float average_number_of_hits_per_event = 0;
  
//   for ( int i_event = 0; i_event < n_events; ++i_event ) {
//     float number_of_hits = 0;
//     const VeloUTTracking::HitsSoA hits_layers = hits_layers_events[i_event];

//     for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
//       debug_cout << "checks on layer " << i_layer << ", with " << hits_layers.n_hits_layers[i_layer] << " hits" << std::endl;
//       number_of_hits += hits_layers.n_hits_layers[i_layer];
//       int layer_offset = hits_layers.layer_offset[i_layer];
//       for ( int i_hit = 0; i_hit < 3; ++i_hit ) {
//         printf("\t at hit %u, cos = %f, yBegin = %f, yEnd = %f, zAtyEq0 = %f, xAtyEq0 = %f, weight = %f, highThreshold = %u, LHCbID = %u, dxDy = %f \n",
//         i_hit,
//         hits_layers.m_cos[ layer_offset + i_hit ],
//         hits_layers.m_yBegin[ layer_offset + i_hit ],
//         hits_layers.m_yEnd[ layer_offset + i_hit ],
//         hits_layers.m_zAtYEq0[ layer_offset + i_hit ],
//         hits_layers.m_xAtYEq0[ layer_offset + i_hit ],
//         hits_layers.m_weight[ layer_offset + i_hit ],
//         hits_layers.m_highThreshold[ layer_offset + i_hit ],
//         hits_layers.m_LHCbID[ layer_offset + i_hit ],
//         hits_layers.dxDy( layer_offset + i_hit ) );
//       }
//     }
    
//     average_number_of_hits_per_event += number_of_hits;
//     debug_cout << "# of UT hits = " << number_of_hits << std::endl;
//   }

void check_scifi_events( const SciFi::HitsSoA *hits_layers_events,
		      const uint32_t n_hits_layers_events[][SciFi::Constants::n_zones],
		      const int n_events ) {

  float average_number_of_hits_per_event = 0;
  
  for ( int i_event = 0; i_event < n_events; ++i_event ) {
    // sanity checks
    float number_of_hits = 0;

    for ( int i_layer = 0; i_layer < SciFi::Constants::n_zones; ++i_layer ) {
      debug_cout << "checks on layer " << i_layer << ", with " << n_hits_layers_events[i_event][i_layer] << " hits" << std::endl;
      number_of_hits += n_hits_layers_events[i_event][i_layer];
      int layer_offset = hits_layers_events[i_event].layer_offset[i_layer];
      for ( int i_hit = 0; i_hit < 3; ++i_hit ) {
	printf("\t at hit %u, x = %f, z = %f, w = %f, dxdy = %f, dzdy = %f, yMin = %f, yMax = %f, LHCbID = %x, planeCode = %i, hitZone = %i \n",
	       i_hit,
	       hits_layers_events[i_event].m_x[ layer_offset + i_hit ],
	       hits_layers_events[i_event].m_z[ layer_offset + i_hit ],
	       hits_layers_events[i_event].m_w[ layer_offset + i_hit ],
	       hits_layers_events[i_event].m_dxdy[ layer_offset + i_hit ],
	       hits_layers_events[i_event].m_dzdy[ layer_offset + i_hit ],
	       hits_layers_events[i_event].m_yMin[ layer_offset + i_hit ],
	       hits_layers_events[i_event].m_yMax[ layer_offset + i_hit ],
               hits_layers_events[i_event].m_LHCbID[ layer_offset + i_hit ],
               hits_layers_events[i_event].m_planeCode[layer_offset + i_hit ],
               hits_layers_events[i_event].m_hitZone[layer_offset + i_hit ] );
      }
    }

    
    average_number_of_hits_per_event += number_of_hits;
    debug_cout << "# of SciFi hits = " << number_of_hits << std::endl;
    
  }

  average_number_of_hits_per_event = average_number_of_hits_per_event / n_events;
  debug_cout << "average # of SciFi hits / event = " << average_number_of_hits_per_event << std::endl;
    
  
}

#define hits_to_out 3
void check_muon_events( const Muon::HitsSoA * muon_station_hits,
  const int n_events ){

  float average_number_of_hits_per_event = 0;

  for ( int i_event = 0; i_event < n_events; ++i_event ) {

    float number_of_hits_per_event = 0;

    for ( int i_station = 0; i_station < Muon::Constants::n_stations; ++i_station ) {

      const int station_offset = muon_station_hits[i_event].m_station_offsets[i_station];
      const int number_of_hits = muon_station_hits[i_event].m_number_of_hits_per_station[i_station];
      number_of_hits_per_event += number_of_hits;

      debug_cout << "checks on station " << i_station << ", with" << number_of_hits << " hits" << std::endl;
      for ( int i_hit; i_hit < hits_to_out; ++i_hit ) {
        printf("\t at hit %u, tile = %i, x = %f, dx = %f, y = %f, dy = %f, z = %f, dz = %f, uncrossed = %i, time = %x, delta_time = %i, cluster_size = %i \n", 
          i_hit,
          muon_station_hits.m_tile[ station_offset + i_hit ],
          muon_station_hits.m_x[ station_offset + i_hit ],
          muon_station_hits.m_dx[ station_offset + i_hit ]
          muon_station_hits.m_y[ station_offset + i_hit ],
          muon_station_hits.m_dy[ station_offset + i_hit ]
          muon_station_hits.m_z[ station_offset + i_hit ],
          muon_station_hits.m_dz[ station_offset + i_hit ],
          muon_station_hits.m_uncrossed[ station_offset + i_hit ],
          muon_station_hits.m_time[ station_offset + i_hit ],
          muon_station_hits.m_delta_time[ station_offset + i_hit ],
          muon_station_hits.m_cluster_size[ station_offset + i_hit ]
          );
      }
    }

    average_number_of_hits_per_event += number_of_hits_per_event;
    debug_cout << "# of Muon hits = " << number_of_hits_per_event << std::endl;
  }

  average_number_of_hits_per_event = average_number_of_hits_per_event / n_events;
  debug_cout << "average # of Muon hits / event = " << average_number_of_hits_per_event << std::endl;
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

void check_roughly(
  const trackChecker::Tracks& tracks,
  const std::vector<uint32_t> hit_IDs,
  const MCParticles mcps
) {
  int matched = 0;
  for ( auto track : tracks ) {
    std::vector< uint32_t > mcp_ids;
    
    for ( LHCbID id : track.ids() ) {
      uint32_t id_int = uint32_t( id );
      // find associated IDs from mcps
      for ( int i_mcp = 0; i_mcp < mcps.size(); ++i_mcp ) {
        MCParticle part = mcps[i_mcp];
        auto it = std::find( part.hits.begin(), part.hits.end(), id_int );
        if ( it != part.hits.end() ) {
          mcp_ids.push_back( i_mcp );
        }
      }
    }
    
    printf("# of hits on track = %u, # of MCP ids = %u \n", track.nIDs(), uint32_t( mcp_ids.size() ) );
    
    for ( int i_id = 0; i_id < mcp_ids.size(); ++i_id ) {
      uint32_t mcp_id = mcp_ids[i_id];
      printf("\t mcp id = %u \n", mcp_id);
      // how many same mcp IDs are there?
      int n_same = count( mcp_ids.begin(), mcp_ids.end(), mcp_id );
      if ( float(n_same) / track.nIDs() >= 0.7 ) {
          matched++;
          break;
      }
    }
  }

  int long_tracks = 0;
  for (auto& part : mcps) {
    if (part.isLong)
      long_tracks++;
  }
  
  printf("efficiency = %f \n", float(matched) / long_tracks );
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
    //debug_cout << "found " << n_scifi_tracks[i_event] << " tracks on GPU " << std::endl;
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
    //debug_cout << "at track " << std::dec << i_track << std::endl;
    for ( int i_hit = 0; i_hit < forward_track.hitsNum; ++i_hit ) {
      //debug_cout<<"\t LHCbIDs Forward["<<i_hit<<"] = " << std::hex << forward_track.LHCbIDs[i_hit]<< std::endl;
      LHCbID lhcb_id( forward_track.LHCbIDs[i_hit] );
      checker_track.addId( lhcb_id );
    }
    checker_tracks.push_back( checker_track );
  }
  //debug_cout<<"end prepareForwardTracks"<<std::endl;

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
    callPrChecker<TrackCheckerVelo> (
      all_tracks,
      folder_name_MC,
      start_event_offset,
      trackType);
  }
  else if ( trackType == "VeloUT" ) {
    callPrChecker<TrackCheckerVeloUT> (
      all_tracks,
      folder_name_MC,
      start_event_offset,
      trackType);
  }
  else if ( trackType == "Forward" ) {
    callPrChecker<TrackCheckerForward> (
      all_tracks,
      folder_name_MC,
      start_event_offset,
      trackType);
  }
  else {
    error_cout << "unknown track type: " << trackType << std::endl;
  }
}

