/** @file Tracks.h
 *
 * @brief SOA Velo Tracks
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-06
 */

#pragma once

#include <array>
#include <cstdint>

#include "LHCbID.h"

namespace trackChecker {
  
  class Track
  {
    
  public:
    SomeLHCbIDs  allids;
    std::size_t n_matched_total = 0;
    float muon_catboost_output;
        
    void addId ( LHCbID id ) {
      allids.push_back(id);
    }
    
    SomeLHCbIDs ids() const {
      return allids;
    }
    
    
  int nIDs() const {
    return allids.size();
  }
  
  };
  
  using Tracks = std::vector< Track >;
}
