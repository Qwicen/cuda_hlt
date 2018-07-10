#include "Tools.h"
#include "../../checker/lib/include/velopix-input-reader.h"
#include "../../checker/lib/include/TrackChecker.h"
#include "../../checker/lib/include/MCParticle.h"

void readGeometry(
  const std::string& foldername,
  std::vector<char>& geometry
) {
  readFileIntoVector(foldername + "/geometry.bin", geometry);
}

void check_velopix_events(
  const std::vector<char> events,
  const std::vector<unsigned int> event_offsets,
  int n_events
) {
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
}

void read_ut_events_into_arrays( VeloUTTracking::HitsSoA *hits_layers_events,
				 uint32_t n_hits_layers_events[][VeloUTTracking::n_layers],
				 const std::vector<char> events,
				 const std::vector<unsigned int> event_offsets,
				 const int n_events ) {

  
  for ( int i_event = 0; i_event < n_events; ++i_event ) {
    const char* raw_input = events.data() + event_offsets[i_event];
    
    /* In the binary input file, the ut hit variables are stored in arrays,
       there are two stations with two layers each,
       one layer is appended to the other
    */
    // first four words: how many hits are in each layer?
    int n_hits_total = 0;
    int accumulated_hits = 0;
    int accumulated_hits_layers[4];
    for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
      n_hits_layers_events[i_event][i_layer] = *((uint32_t*)raw_input);
      n_hits_total += n_hits_layers_events[i_event][i_layer];
      raw_input += sizeof(uint32_t);
      assert( n_hits_total < VeloUTTracking::max_numhits_per_event );
      hits_layers_events[i_event].layer_offset[i_layer] = accumulated_hits;
      accumulated_hits += n_hits_layers_events[i_event][i_layer];
    }
    // then the hit variables, sorted by layer
    for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
      int layer_offset = hits_layers_events[i_event].layer_offset[i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &( hits_layers_events[i_event].m_cos[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].m_yBegin[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].m_yEnd[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      // to do: change tracker dumper to not dump dxDy
      //std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].m_dxDy[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].m_zAtYEq0[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].m_xAtYEq0[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].m_weight2[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((int*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].m_highThreshold[ layer_offset ]) );
      raw_input += sizeof(int) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((unsigned int*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].m_LHCbID[ layer_offset ]) );
      raw_input += sizeof(unsigned int) * n_hits_layers_events[i_event][i_layer];

      for ( int i_hit = 0; i_hit < n_hits_layers_events[i_event][i_layer]; ++i_hit ) {
	hits_layers_events[i_event].m_planeCode[ layer_offset + i_hit ] = i_layer;
      }
    }
 
  
  }
}


void check_ut_events( const VeloUTTracking::HitsSoA *hits_layers_events,
		      const uint32_t n_hits_layers_events[][VeloUTTracking::n_layers],
		      const int n_events ) {

  float average_number_of_hits_per_event = 0;
  
  for ( int i_event = 0; i_event < n_events; ++i_event ) {
    // sanity checks
    float number_of_hits = 0;

    for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
      debug_cout << "checks on layer " << i_layer << ", with " << n_hits_layers_events[i_event][i_layer] << " hits" << std::endl;
      number_of_hits += n_hits_layers_events[i_event][i_layer];
      int layer_offset = hits_layers_events[i_event].layer_offset[i_layer];
      for ( int i_hit = 0; i_hit < 3; ++i_hit ) {
	printf("\t at hit %u, cos = %f, yBegin = %f, yEnd = %f, zAtyEq0 = %f, xAtyEq0 = %f, weight2 = %f, highThreshold = %u, LHCbID = %u, dxDy = %f \n",
	       i_hit,
	       hits_layers_events[i_event].m_cos[ layer_offset + i_hit ],
	       hits_layers_events[i_event].m_yBegin[ layer_offset + i_hit ],
	       hits_layers_events[i_event].m_yEnd[ layer_offset + i_hit ],
	       hits_layers_events[i_event].m_zAtYEq0[ layer_offset + i_hit ],
	       hits_layers_events[i_event].m_xAtYEq0[ layer_offset + i_hit ],
	       hits_layers_events[i_event].m_weight2[ layer_offset + i_hit ],
	       hits_layers_events[i_event].m_highThreshold[ layer_offset + i_hit ],
               hits_layers_events[i_event].m_LHCbID[ layer_offset + i_hit ],
               hits_layers_events[i_event].dxDy( layer_offset + i_hit ) );
      }
    }

    
    average_number_of_hits_per_event += number_of_hits;
    debug_cout << "# of UT hits = " << number_of_hits << std::endl;
    
  }

  average_number_of_hits_per_event = average_number_of_hits_per_event / n_events;
  debug_cout << "average # of UT hits / event = " << average_number_of_hits_per_event << std::endl;
    
  
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

/**
 * @brief Writes a track in binary format
 * 
 * @details The binary format is per every track:
 *   hitsNum hit0 hit1 hit2 ... (#hitsNum times)
 */
template <bool mc_check>
void writeBinaryTrack(
  const unsigned int* hit_IDs,
  const VeloTracking::Track <mc_check> & track,
  std::ofstream& outstream
) {
  uint32_t hitsNum = track.hitsNum;
  outstream.write((char*) &hitsNum, sizeof(uint32_t));
  for (int i=0; i<track.hitsNum; ++i) {
    const VeloTracking::Hit <mc_check> hit = track.hits[i];
    outstream.write((char*) &hit.LHCbID, sizeof(uint32_t));
  }
}

/**
 * Prints tracks
 * Track #n, length <length>:
 *  <ID> module <module>, x <x>, y <y>, z <z>
 * 
 * @param tracks      
 * @param trackNumber 
 */
template <bool mc_check>
void printTrack(
  VeloTracking::Track <mc_check> * tracks,
  const int trackNumber,
  std::ofstream& outstream
) {
  const VeloTracking::Track<mc_check> t = tracks[trackNumber];
  outstream << "Track #" << trackNumber << ", length " << (int) t.hitsNum << std::endl;

  for(int i=0; i<t.hitsNum; ++i){
    const VeloTracking::Hit <mc_check> hit = t.hits[i];
    const float x = hit.x;
    const float y = hit.y;
    const float z = hit.z;
    //const uint LHCbID = hit.LHCbID;
  
    outstream 
      << ", x " << std::setw(6) << x
      << ", y " << std::setw(6) << y
      << ", z " << std::setw(6) << z
      //<< ", LHCbID " << LHCbID 
      << std::endl;
  }

  outstream << std::endl;
}

template <bool mc_check>
void printTracks(
  VeloTracking::Track <mc_check> * tracks,
  int* n_tracks,
  int n_events,
  std::ofstream& outstream 
) {
  for ( int i_event = 0; i_event < n_events; ++i_event ) {
    VeloTracking::Track <mc_check> * tracks_event = tracks + i_event * VeloTracking::max_tracks;
    int n_tracks_event = n_tracks[ i_event ];
    outstream << i_event << std::endl;
    outstream << n_tracks_event << std::endl;

    for ( int i_track = 0; i_track < n_tracks_event; ++i_track ) {
      const VeloTracking::Track <mc_check> t = tracks_event[i_track];
      outstream << t.hitsNum << std::endl;
      for ( int i_hit = 0; i_hit < t.hitsNum; ++i_hit ) {
        VeloTracking::Hit <mc_check> hit = t.hits[ i_hit ];
        outstream << hit.LHCbID << std::endl;
      }
    }
  }
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
        auto it = std::find( part.m_hits.begin(), part.m_hits.end(), id_int );
        if ( it != part.m_hits.end() ) {
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
    if (part.isLong())
      long_tracks++;
  }
  
  printf("efficiency = %f \n", float(matched) / long_tracks );
}

std::vector< trackChecker::Tracks > prepareTracks(
  uint* host_velo_track_hit_number_pinned,
  VeloTracking::Hit<true>* host_velo_track_hits_pinned,  
  int* host_accumulated_tracks,
  int* host_number_of_tracks_pinned,
  const int &number_of_events
) {
  
  /* Tracks to be checked, save in format for checker */
  std::vector< trackChecker::Tracks > all_tracks; // all tracks from all events
  for ( uint i_event = 0; i_event < number_of_events; i_event++ ) {
    trackChecker::Tracks tracks; // all tracks within one event
    const int accumulated_tracks = host_accumulated_tracks[i_event];
    for ( uint i_track = 0; i_track < host_number_of_tracks_pinned[i_event]; i_track++ ) {
      trackChecker::Track t;
      const uint starting_hit = host_velo_track_hit_number_pinned[accumulated_tracks + i_track];
      const uint number_of_hits = host_velo_track_hit_number_pinned[accumulated_tracks + i_track + 1] - starting_hit;
      
      for ( int i_hit = 0; i_hit < number_of_hits; ++i_hit ) {
        t.addId(host_velo_track_hits_pinned[starting_hit + i_hit].LHCbID);
      } 
      tracks.push_back( t );
    } // tracks
    all_tracks.emplace_back( tracks );
  }
  
  return all_tracks;
}


trackChecker::Tracks prepareVeloUTTracks(
  std::vector< VeloUTTracking::TrackUT > ut_tracks
) {
  trackChecker::Tracks checker_tracks;
  for ( VeloUTTracking::TrackUT ut_track : ut_tracks ) {
    trackChecker::Track checker_track;
    for ( int i_hit = 0; i_hit < ut_track.hitsNum; ++i_hit ) {
      LHCbID lhcb_id( ut_track.LHCbIDs[i_hit] );
      checker_track.addId( lhcb_id );
    }
    checker_tracks.push_back( checker_track );
  }

  return checker_tracks;
}


void call_pr_checker(
  const std::vector< trackChecker::Tracks >& all_tracks,
  const std::string& folder_name_MC,
  const bool& fromNtuple,
  const std::string& trackType
) {
  if ( trackType == "Velo" ) {
    callPrChecker< TrackCheckerVelo> (
      all_tracks,
      folder_name_MC,
      fromNtuple,
      trackType);
  }
  else if ( trackType == "VeloUT" ) {
    callPrChecker< TrackCheckerVeloUT> (
      all_tracks,
      folder_name_MC,
      fromNtuple,
      trackType);
  }
  else {
    error_cout << "unknown track type: " << trackType << std::endl;
  }
}

