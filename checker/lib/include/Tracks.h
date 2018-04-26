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

#include "boost/range/iterator_range.hpp"

#include "LHCbID.h"


class Track
{

 public:
  SomeLHCbIDs  allids;
  
  void addId ( LHCbID id ) {
    allids.push_back(id);
  }
  
  SomeLHCbIDs ids() const {
    //SomeLHCbIDs id_collection = SomeLHCbIDs( allids.begin(), allids.end() );
    //return id_collection;
    return allids;
  }

 
  int nIDs() const {
    return allids.size();
  }

  
};

using Tracks = std::vector< Track >;

//using Tracks = SOA::Container<std::vector, TracksDesc::Skin>;
//using TracksRange = boost::iterator_range<Tracks::iterator>;

// vim: sw=4:tw=78:ft=cpp:et
