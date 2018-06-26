#include "../include/Tools.h"
#include "../../checker/lib/include/velopix-input-reader.h"
#include "../../checker/lib/include/TrackChecker.h"
#include "../../checker/lib/include/MCParticle.h"
#include "../../checker/lib/include/velopix-input-reader.h"

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

    //printf("at event %u, # of raw banks = %u, first offset = %u, second offset = %u \n", i_event, number_of_raw_banks, raw_bank_offset[0], raw_bank_offset[1]);
    
    uint32_t sensor =  *((uint32_t*)p);  p += sizeof(uint32_t);
    uint32_t sp_count =  *((uint32_t*)p); p += sizeof(uint32_t);
    //printf("sensor = %u, sp_count = %u \n", sensor, sp_count);

    const auto raw_event = VeloRawEvent(raw_input);
    int n_sps_event = 0;
    for ( int i_raw_bank = 0; i_raw_bank < raw_event.number_of_raw_banks; i_raw_bank++ ) {
      const auto raw_bank = VeloRawBank(raw_event.payload + raw_event.raw_bank_offset[i_raw_bank]);
      n_sps_event += raw_bank.sp_count;
      if ( i_raw_bank != raw_bank.sensor_index ) {
        error_cout << "at raw bank " << i_raw_bank << ", but index = " << raw_bank.sensor_index << std::endl;
      }
      //printf("\t sensor = %u, sp_count = %u \n",  raw_bank.sensor_index, raw_bank.sp_count);
      if ( raw_bank.sp_count > 0 ) {
        uint32_t sp_word = raw_bank.sp_word[0];
        uint8_t sp = sp_word & 0xFFU;
        if (0 == sp) { continue; };
        const uint32_t sp_addr = (sp_word & 0x007FFF00U) >> 8;
        const uint32_t sp_row = sp_addr & 0x3FU;
        const uint32_t sp_col = (sp_addr >> 6);
        const uint32_t no_sp_neighbours = sp_word & 0x80000000U;
        //printf("\t first sp col = %u, row = %u \n", sp_col, sp_row);
      }
      
    }
    n_sps_all_events += n_sps_event;
    //printf("# of sps in this event = %u\n", n_sps_event);
  }

  // printf("total # of sps = %u \n", n_sps_all_events);
  // float n_sps_average = (float)n_sps_all_events / n_events;
  // printf("average # of sps per event = %f \n", n_sps_average);
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
      accumulated_hits_layers[i_layer] = accumulated_hits;
      accumulated_hits += n_hits_layers_events[i_event][i_layer];
    }
    // then the hit variables, sorted by layer
    for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
      int layer_offset = accumulated_hits_layers[i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &( hits_layers_events[i_event].cos[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].yBegin[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].yEnd[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].dxDy[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].zAtYEq0[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].xAtYEq0[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((float*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].weight[ layer_offset ]) );
      raw_input += sizeof(float) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((int*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].highThreshold[ layer_offset ]) );
      raw_input += sizeof(int) * n_hits_layers_events[i_event][i_layer];
      std::copy_n((unsigned int*) raw_input, n_hits_layers_events[i_event][i_layer], &(hits_layers_events[i_event].LHCbID[ layer_offset ]) );
      raw_input += sizeof(unsigned int) * n_hits_layers_events[i_event][i_layer];
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

    // find out offsets for every layer
    int accumulated_hits = 0;
    int accumulated_hits_layers[4];
    for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
      accumulated_hits_layers[i_layer] = accumulated_hits;
      accumulated_hits += n_hits_layers_events[i_event][i_layer];
    }
    for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
      debug_cout << "checks on layer " << i_layer << ", with " << n_hits_layers_events[i_event][i_layer] << " hits" << std::endl;
      number_of_hits += n_hits_layers_events[i_event][i_layer];
      for ( int i_hit = 0; i_hit < 3; ++i_hit ) {
	int layer_offset = accumulated_hits_layers[i_layer];
	printf("\t at hit %u, cos = %f, yBegin = %f, yEnd = %f, dxDy = %f, zAtyEq0 = %f, xAtyEq0 = %f, weight = %f, highThreshold = %u, LHCbID = %u \n",
	       i_hit,
	       hits_layers_events[i_event].cos[ layer_offset + i_hit ],
	       hits_layers_events[i_event].yBegin[ layer_offset + i_hit ],
	       hits_layers_events[i_event].yEnd[ layer_offset + i_hit ],
	       hits_layers_events[i_event].dxDy[ layer_offset + i_hit ],
	       hits_layers_events[i_event].zAtYEq0[ layer_offset + i_hit ],
	       hits_layers_events[i_event].xAtYEq0[ layer_offset + i_hit ],
	       hits_layers_events[i_event].weight[ layer_offset + i_hit ],
	       hits_layers_events[i_event].highThreshold[ layer_offset + i_hit ],
               hits_layers_events[i_event].LHCbID[ layer_offset + i_hit ] );
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
  const std::vector<VelopixEvent::MCP> mcps
) {
  int matched = 0;
  for ( auto track : tracks ) {
    std::vector< uint32_t > mcp_ids;
    
    for ( LHCbID id : track.ids() ) {
      uint32_t id_int = uint32_t( id );
      // find associated IDs from mcps
      for ( int i_mcp = 0; i_mcp < mcps.size(); ++i_mcp ) {
        VelopixEvent::MCP part = mcps[i_mcp];
        auto it = find( part.hits.begin(), part.hits.end(), id_int );
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
  for ( VelopixEvent::MCP part : mcps ) {
    if ( part.islong )
      long_tracks++;
  }
  
  printf("efficiency = %f \n", float(matched) / long_tracks );
}

void callPrChecker(
  const std::vector< trackChecker::Tracks >& all_tracks,
  const std::string& folder_name_MC,
  const bool& fromNtuple,
  const std::string& trackType
) {

  /* MC information */
  int n_events = all_tracks.size();
    std::vector<VelopixEvent> events = read_mc_folder(folder_name_MC, fromNtuple, trackType, n_events, true );
  
  TrackChecker trackChecker {};
  uint64_t evnum = 0; // DvB: check, was 1 before!!

  for (const auto& ev: events) {
    debug_cout << "Event " << (evnum+1) << std::endl;
    const auto& mcps = ev.mcparticles();
    const std::vector<uint32_t>& hit_IDs = ev.hit_IDs;
    const std::vector<VelopixEvent::MCP>& mcps_vector = ev.mcps;
    MCAssociator mcassoc(mcps);

    debug_cout << "Found " << all_tracks[evnum].size() << " reconstructed tracks" <<
     " and " << mcps.size() << " MC particles " << std::endl;

    trackChecker(all_tracks[evnum], mcassoc, mcps);
    //check_roughly(tracks, hit_IDs, mcps_vector);

    ++evnum;
  }
}

std::vector< trackChecker::Tracks > prepareTracks(
  VeloTracking::Track <true> * host_tracks_pinned,
  int * host_accumulated_tracks,
  int * host_number_of_tracks_pinned,
  const int &number_of_events
) {
  
  /* Tracks to be checked, save in format for checker */
  std::vector< trackChecker::Tracks > all_tracks; // all tracks from all events
  for ( uint i_event = 0; i_event < number_of_events; i_event++ ) {
    VeloTracking::Track<true>* tracks_event = host_tracks_pinned + host_accumulated_tracks[i_event];
    trackChecker::Tracks tracks; // all tracks within one event
    
    for ( uint i_track = 0; i_track < host_number_of_tracks_pinned[i_event]; i_track++ ) {
      trackChecker::Track t;
      const VeloTracking::Track<true> track = tracks_event[i_track];
      
      for ( int i_hit = 0; i_hit < track.hitsNum; ++i_hit ) {
        VeloTracking::Hit<true> hit = track.hits[ i_hit ];
        LHCbID lhcb_id( hit.LHCbID );
        t.addId( lhcb_id );
      } // hits
      tracks.push_back( t );
    } // tracks
    
    all_tracks.emplace_back( tracks );
  } // events

  return all_tracks;
}
