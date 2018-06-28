#pragma once

#include <string>
#include <cstdint>
#include <vector>

#include "VeloPixels.h"
#include "MCParticle.h"

#include "../../../main/include/Common.h"
#include "../../../main/include/Logger.h"
#include "../../../main/include/InputTools.h"
#include "TrackChecker.h"

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

    // Event data
    uint32_t numberOfModules;
    uint32_t numberOfHits;
    std::vector<float> module_Zs;
    std::vector<uint32_t> module_hitStarts;
    std::vector<uint32_t> module_hitNums;
    std::vector<uint32_t> hit_IDs;
    std::vector<float> hit_Xs;
    std::vector<float> hit_Ys;
    std::vector<float> hit_Zs;

    // Monte Carlo information
    struct MCP {
        uint32_t key;
        uint32_t id;
        float p;
        float pt;
        float eta;
        float phi;
        bool islong;
        bool isdown;
        bool isvelo;
        bool isut;
        bool strangelong;
        bool strangedown;
        bool fromb;
        bool fromd;
        uint32_t numHits;
        std::vector<uint32_t> hits;
    };

    std::vector<MCP> mcps;

    // Constructor
    VelopixEvent() {};
    VelopixEvent(const std::vector<char>& _event, const bool checkFile = true);

    void print() const;

    /// get hits into a format we like (should ultimately go away, or just be a view)
    VeloPixels soaHits() const;
    MCParticles mcparticles() const;
};

void readNtupleIntoVelopixEvent(const std::string& filename, const std::string& trackType, VelopixEvent& event);
 
std::vector<VelopixEvent> read_mc_folder(
  const std::string& foldername,
  const bool& fromNtuple,
  const std::string& trackType,
  uint number_of_files,
  const bool checkEvents = false
);
 
template< typename t_checker >
void callPrChecker(
  const std::vector< trackChecker::Tracks >& all_tracks,
  const std::string& folder_name_MC,
  const bool& fromNtuple,
  const std::string& trackType
) {
   /* MC information */
  int n_events = all_tracks.size();
    std::vector<VelopixEvent> events = read_mc_folder(folder_name_MC, fromNtuple, trackType, n_events, true );

    
  t_checker trackChecker {};
  uint64_t evnum = 0; 

  for (const auto& ev: events) {
    const auto& mcps = ev.mcparticles();
    const std::vector<VelopixEvent::MCP>& mcps_vector = ev.mcps;
    MCAssociator mcassoc(mcps);

    trackChecker(all_tracks[evnum], mcassoc, mcps);

    // /* Check for double counting of hits */
    // for ( auto ch_track : all_tracks[evnum] ) {
    //   for ( auto ida : ch_track.ids() ) {
    // 	int counted_IDs = 0;
    // 	for ( auto idb : ch_track.ids() ) {
    // 	  if ( uint32_t(ida) == uint32_t(idb) )
    // 	    counted_IDs++;
    // 	}

    // 	if ( counted_IDs > 1  ) {
    // 	  warning_cout << "ATTENTION: counted " << counted_IDs << " same IDs on one track" << std::endl;
    // 	  for ( auto id : ch_track.ids() ) {
    // 	    warning_cout << std::hex << "\t " << id << std::endl;
    // 	  }
    // 	}
	
    //   }
    // }
		
    
    ++evnum;
  }
}
