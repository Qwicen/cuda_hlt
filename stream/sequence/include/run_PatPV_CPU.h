#pragma once

#include "../../../main/include/Common.h"

#include "../../../main/include/Tools.h"


#include "../../../PatPV/include/PVSeedTool.h"
#include "../../../PatPV/include/AdaptivePV3DFitter.h"

#include "TH1D.h"
#include "TFile.h"
#include "TTree.h"
#include <algorithm>

/*
template <typename C>
void removeTracks(std::vector< Track*>& tracks,
                                  C& tracks2remove) 
{


  for( Track* trk : tracks2remove) {
    auto i = std::find(tracks.begin(),tracks.end(), trk);
    if ( i != tracks.end()) tracks.erase(i);
  }


}
*/

int run_PatPV_on_CPU (
  VeloState * host_velo_states,
  int * host_accumulated_tracks,
  uint* host_velo_track_hit_number_pinned,
  VeloTracking::Hit<true>* host_velo_track_hits_pinned,
  int * host_number_of_tracks_pinned,
  const int &number_of_events,
  Vertex * outvtxvec,
  uint * number_of_vertex
);

