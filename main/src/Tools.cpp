#include "../include/Tools.h"

#include "../../checker/lib/include/velopix-input-reader.h"
#include "../../checker/lib/include/TrackChecker.h"

#include "../../checker/lib/include/MCParticle.h"

/**
 * @brief Read files into vectors.
 */
void readFileIntoVector(
  const std::string& filename,
  std::vector<char>& events
) {
  std::ifstream infile(filename.c_str(), std::ifstream::binary);
  infile.seekg(0, std::ios::end);
  auto end = infile.tellg();
  infile.seekg(0, std::ios::beg);
  auto dataSize = end - infile.tellg();

  events.resize(dataSize);
  infile.read((char*) &(events[0]), dataSize);
  infile.close();
}

/**
 * @brief Appends a file to a vector of char.
 *        It can also be used to read a file to a vector,
 *        returning the event_size.
 */
void appendFileToVector(
  const std::string& filename,
  std::vector<char>& events,
  std::vector<unsigned int>& event_sizes
) {
  std::ifstream infile(filename.c_str(), std::ifstream::binary);
  infile.seekg(0, std::ios::end);
  auto end = infile.tellg();
  infile.seekg(0, std::ios::beg);
  auto dataSize = end - infile.tellg();

  // read content of infile with a vector
  const size_t previous_size = events.size();
  events.resize(events.size() + dataSize);
  infile.read(events.data() + previous_size, dataSize);
  event_sizes.push_back(dataSize);
  infile.close();

}

void readGeometry(
  const std::string& foldername,
  std::vector<char>& geometry
) {
  readFileIntoVector(foldername + "/geometry.bin", geometry);
}

void check_events( const std::vector<char> events,
		   const std::vector<unsigned int> event_offsets,
		   int n_events) {

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
      if ( i_raw_bank != raw_bank.sensor_index )
	printf("ERROR: at raw bank %u, but index = %u \n", i_raw_bank, raw_bank.sensor_index);
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

  printf("total # of sps = %u \n", n_sps_all_events);
  
}

/**
 * @brief Reads a number of events from a folder name.
 */
void readFolder(
  const std::string& foldername,
  unsigned int number_of_files,
  std::vector<char>& events,
  std::vector<unsigned int>& event_offsets
) {
  std::vector<std::string> folderContents;
  DIR *dir;
  struct dirent *ent;

  // Find out folder contents
  if ((dir = opendir(foldername.c_str())) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL) {
      std::string filename = std::string(ent->d_name);
      if (filename.find(".bin") != std::string::npos &&
        filename.find("geometry") == std::string::npos) {
        folderContents.push_back(filename);
      }
    }
    closedir(dir);
    if (folderContents.size() == 0) {
      std::cerr << "No binary files found in folder " << foldername << std::endl;
      exit(-1);
    } else {
      std::cout << "Found " << folderContents.size() << " binary files" << std::endl;
    }
  } else {
    std::cerr << "Folder could not be opened" << std::endl;
    exit(-1);
  }

  // Sort folder contents (file names)
  std::sort(folderContents.begin(), folderContents.end()); 

  if (number_of_files == 0) {
    number_of_files = folderContents.size();
  }

  std::cout << "Requested " << number_of_files << " files" << std::endl;
  int readFiles = 0;

  // Read all requested events
  unsigned int accumulated_size=0;
  std::vector<unsigned int> event_sizes;
  for (int i=0; i<number_of_files; ++i) {
    // Read event #i in the list and add it to the inputs
    std::string readingFile = folderContents[i % folderContents.size()];
    appendFileToVector(foldername + "/" + readingFile, events, event_sizes);
    std::cout << "Reading raw event " <<  readingFile << std::endl;

    event_offsets.push_back(accumulated_size);
    accumulated_size += event_sizes.back();
    
    readFiles++;
    if ((readFiles % 100) == 0) {
      std::cout << "." << std::flush;
    }
  }

  // Add last offset
  event_offsets.push_back(accumulated_size);

  std::cout << std::endl << (event_offsets.size() - 1) << " files read" << std::endl << std::endl;

  check_events( events, event_offsets, number_of_files );
}

/**
 * @brief Print statistics from the input files
 */
void statistics(
  const std::vector<char>& input,
  std::vector<unsigned int>& event_offsets
) {

  // unsigned int max_number_of_hits = 0;
  // unsigned int max_number_of_hits_in_module = 0;
  // unsigned int average_number_of_hits_in_module = 0;

  // for (size_t i=0; i<event_offsets.size(); ++i) {
  //   EventInfo info (input.data() + event_offsets[i]);
  //   for (size_t j=0; j<info.numberOfModules; ++j) {
  //     max_number_of_hits_in_module = std::max(max_number_of_hits_in_module, info.module_hitNums[j]);
  //     average_number_of_hits_in_module += info.module_hitNums[j];
  //   }
  //   max_number_of_hits = std::max(max_number_of_hits, info.numberOfHits);
  // }
  // average_number_of_hits_in_module /= event_offsets.size() * N_MODULES;

  // std::cout << "Statistics on input events:" << std::endl
  //   << " Max number of hits in event: " << max_number_of_hits << std::endl
  //   << " Max number of hits in one module: " << max_number_of_hits_in_module << std::endl
  //   << " Average number of hits in module: " << average_number_of_hits_in_module << std::endl << std::endl;
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
  const Track <mc_check> & track,
  std::ofstream& outstream
) {
  uint32_t hitsNum = track.hitsNum;
  outstream.write((char*) &hitsNum, sizeof(uint32_t));
  for (int i=0; i<track.hitsNum; ++i) {
    const Hit <true> hit = track.hits[i];
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
  Track <mc_check> * tracks,
  const int trackNumber,
  std::ofstream& outstream
) {
  const Track<mc_check> t = tracks[trackNumber];
  outstream << "Track #" << trackNumber << ", length " << (int) t.hitsNum << std::endl;

  for(int i=0; i<t.hitsNum; ++i){
    const Hit <mc_check> hit = t.hits[i];
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
  Track <mc_check> * tracks,
  int* n_tracks,
  int n_events,
  std::ofstream& outstream 
) {
  for ( int i_event = 0; i_event < n_events; ++i_event ) {
    Track <mc_check> * tracks_event = tracks + i_event * VeloTracking::max_tracks;
    int n_tracks_event = n_tracks[ i_event ];
    outstream << i_event << std::endl;
    outstream << n_tracks_event << std::endl;

    for ( int i_track = 0; i_track < n_tracks_event; ++i_track ) {
      const Track <mc_check> t = tracks_event[i_track];
      outstream << t.hitsNum << std::endl;
      for ( int i_hit = 0; i_hit < t.hitsNum; ++i_hit ) {
	Hit <true> hit = t.hits[ i_hit ];
	outstream << hit.LHCbID << std::endl;
      }
    }
  }
  
}

void printOutAllModuleHits(const EventInfo& info, int* prevs, int* nexts) {
  std::cout << "All valid module hits: " << std::endl;
  for(int i=0; i<info.numberOfModules; ++i){
    for(int j=0; j<info.module_hitNums[i]; ++j){
      int hit = info.module_hitStarts[i] + j;
 
      if(nexts[hit] != -1){
	std::cout << hit << ", " << nexts[hit] << std::endl;
      }
    }
  }
}

void printOutModuleHits(const EventInfo& info, int moduleNumber, int* prevs, int* nexts){
  for(int i=0; i<info.module_hitNums[moduleNumber]; ++i){
    int hstart = info.module_hitStarts[moduleNumber];

    std::cout << hstart + i << ": " << prevs[hstart + i] << ", " << nexts[hstart + i] << std::endl;
  }
}

void printInfo(const EventInfo& info, int numberOfModules, int numberOfHits) {
  numberOfModules = numberOfModules>N_MODULES ? N_MODULES : numberOfModules;

  std::cout << "Read info:" << std::endl
    << " no modules: " << info.numberOfModules << std::endl
    << " no hits: " << info.numberOfHits << std::endl
    << numberOfModules << " modules: " << std::endl;

  for (int i=0; i<numberOfModules; ++i){
    std::cout << " Zs: " << info.module_Zs[i] << std::endl
      << " hitStarts: " << info.module_hitStarts[i] << std::endl
      << " hitNums: " << info.module_hitNums[i] << std::endl << std::endl;
  }

  std::cout << numberOfHits << " hits: " << std::endl;

  for (int i=0; i<numberOfHits; ++i){
    std::cout << " hit_id: " << info.hit_IDs[i] << std::endl
      << " hit_X: " << info.hit_Xs[i] << std::endl
      << " hit_Y: " << info.hit_Ys[i] << std::endl
      // << " hit_Z: " << info.hit_Zs[i] << std::endl
      << std::endl;
  }
}

void check_roughly( const trackChecker::Tracks& tracks, const std::vector<uint32_t> hit_IDs, const std::vector< VelopixEvent::MCP > mcps ) {

  int matched = 0;
  for ( auto track : tracks ) {
    std::vector< uint32_t > mcp_ids;
    
    for ( LHCbID id : track.ids() ) {
      uint32_t id_int = uint32_t( id );
      
      //for ( uint32_t id_int : hit_IDs ) {
	
      // find associated IDs from mcps
      for ( int i_mcp = 0; i_mcp < mcps.size(); ++i_mcp ) {
	VelopixEvent::MCP part = mcps[i_mcp];
	auto it = find( part.hits.begin(), part.hits.end(), id_int );
	if ( it != part.hits.end() ) {
	  mcp_ids.push_back( i_mcp );
	  //matched++;
	}
      }
    }
    
    //printf("# of hits on track = %u, # of MCP ids = %u \n", track.nIDs(), uint32_t( mcp_ids.size() ) );
    
    for ( int i_id = 0; i_id < mcp_ids.size(); ++i_id ) {
      uint32_t mcp_id = mcp_ids[i_id];
      //printf("\t mcp id = %u p = %f \n", mcp_id);
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
  
      // int n_mcp_ids = 0;
      // for ( int i_mcp = 0; i_mcp < mcps.size(); ++i_mcp ) {
      // 	VelopixEvent::MCP part = mcps[i_mcp];
      // 	n_mcp_ids += part.numHits;
      // }
      
      // printf("single id efficiency = %f \n", float(matched) / n_mcp_ids);
      
}

void checkTracks( const std::vector< trackChecker::Tracks > all_tracks, const std::string folder_name_MC ) {

  /* MC information */
  int n_events = all_tracks.size();
  std::vector<VelopixEvent> events = VelopixEventReader::readFolder(folder_name_MC, n_events, true );
  
  TrackChecker trackChecker;
  uint64_t evnum = 1;
  for (const auto& ev: events) {
    std::cout << "Event " << evnum << std::endl;
    auto mcps = ev.mcparticles();
    std::vector< uint32_t > hit_IDs = ev.hit_IDs;
    std::vector<VelopixEvent::MCP> mcps_vector = ev.mcps;
    MCAssociator mcassoc(mcps);

    trackChecker::Tracks tracks = all_tracks[evnum-1];
    std::cout << "INFO: found " << tracks.size() << " reconstructed tracks" <<
     " and " << mcps.size() << " MC particles " << std::endl;

    trackChecker(tracks, mcassoc, mcps);
    check_roughly(tracks, hit_IDs, mcps_vector);
        
    ++evnum;
  }
  
}
