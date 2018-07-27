/** @file velopix-input-reader.h
 *
 * @brief a reader of velopix inputs
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-18
 *
 * 2018-07 Dorothea vom Bruch: updated to run over different track types, 
 * take input from Renato Quagliani's TrackerDumper
 */

#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include <algorithm>

#include "MCParticle.h"

#include "../../../main/include/Common.h"
#include "../../../main/include/Logger.h"
#include "../../../main/include/InputTools.h"
#include "TrackChecker.h"
#include "MCParticle.h"

class VelopixEvent {
private:
    template<class T>
    static std::string strVector(const T v, const uint vSize, const uint numberOfElements = 5) {
        std::string s = "";
        auto n = std::min(vSize, numberOfElements);
        for (size_t i=0; i<n; ++i) {
            s += std::to_string(v[i]);
            if (i != n-1) s += ", ";
            else if (i == vSize-1) s += "";
            else s += "...";
        }
        return s;
    }

public:
    uint32_t size;
  MCParticles mcps;

    // Constructor
    VelopixEvent() {};
    VelopixEvent(const std::vector<char>& _event, const std::string& trackType, const bool checkFile = true);

    void print() const;

    MCParticles mcparticles() const;
};

std::vector<VelopixEvent> read_mc_folder(
  const std::string& foldername,
  const std::string& trackType,
  uint number_of_files,
  const uint start_event_offset,
  const bool checkEvents = false
);
 
template< typename t_checker >
void callPrChecker(
  const std::vector< trackChecker::Tracks >& all_tracks,
  const std::string& folder_name_MC,
  const uint start_event_offset,
  const std::string& trackType
) {
   /* MC information */
  int n_events = all_tracks.size();
  std::vector<VelopixEvent> events = read_mc_folder(folder_name_MC, trackType, n_events, start_event_offset, true );

    
  t_checker trackChecker {};
  uint64_t evnum = 0; 

  for (const auto& ev: events) {
    const auto& mcps = ev.mcparticles();
    const std::vector<MCParticle>& mcps_vector = ev.mcps;
    MCAssociator mcassoc(mcps);

    trackChecker(all_tracks[evnum], mcassoc, mcps);

    /* Check for double counting of hits */
    uint i_track = 0;
    for ( auto ch_track : all_tracks[evnum] ) {
      for ( uint i_a = 0; i_a < ch_track.nIDs(); ++i_a ) {
        auto ida = ch_track.ids()[i_a];
    	int counted_IDs = 0;
        for ( uint i_b = i_a; i_b < ch_track.nIDs(); ++i_b ) {
          auto idb = ch_track.ids()[i_b];
    	  if ( uint32_t(ida) == uint32_t(idb) )
    	    counted_IDs++;
    	}

    	if ( counted_IDs > 1  ) {
    	  warning_cout << "ATTENTION: counted " << counted_IDs << " same IDs on track # " << std::dec << i_track << std::endl;
    	  for ( auto id : ch_track.ids() ) {
    	    warning_cout << std::hex << "\t " << id << std::endl;
    	  }
    	}
	
      }
      ++i_track;
    }
		
    
    ++evnum;
  }
}
