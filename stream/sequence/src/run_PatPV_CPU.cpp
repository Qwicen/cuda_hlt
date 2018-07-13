#include "../include/run_PatPV_CPU.h"
#include "../../../PatPV/include/PVSeedTool.h"
#include "../../../PatPV/include/AdaptivePV3DFitter.h"

#include "TH1D.h"
#include "TFile.h"
#include "TTree.h"
#include <algorithm>


/*
XYZPoint& seedPoint,
              std::vector<Track*>& rTracks,
             Vertex& vtx,
             std::vector<Track*>& tracks2remove

*/





bool reconstructMultiPVFromTracks( std::vector< VeloState>& tracks2use,
                                                       std::vector<Vertex>& outvtxvec, int host_number_of_tracks_pinned) 
{
  

  auto rtracks = tracks2use;

  outvtxvec.clear();


  PVSeedTool seedtool;
  double m_beamSpotX = 0.02;
  double m_beamSpotY = -0.16;
  XYZPoint beamspot{m_beamSpotX, m_beamSpotY, 0.};
  

    


  int nvtx_before = -1;
  int nvtx_after  =  0;
  //for (int i = 0; i < 5 ; i++) {
  while ( nvtx_after > nvtx_before ) {
    nvtx_before = outvtxvec.size();
    // reconstruct vertices


  AdaptivePV3DFitter fitter;
  std::vector<XYZPoint> seeds = seedtool.getSeeds(rtracks.data(), beamspot, host_number_of_tracks_pinned);
    for ( auto seed : seeds) {
      Vertex recvtx;


      std::vector< VeloState> tracks2remove;
      
      // fitting
      bool scvfit = fitter.fitVertex( seed, rtracks.data(), recvtx, tracks2remove, host_number_of_tracks_pinned);
      if (!scvfit) continue;
      
      

      
      outvtxvec.push_back(recvtx);
      //removeTracks(rtracks, tracks2remove);
    }//iterate on seeds
    nvtx_after = outvtxvec.size();
  }//iterate on vtx

  return true;

}




int run_PatPV_on_CPU (
  VeloState * host_velo_states,
  int * host_accumulated_tracks,
  uint* host_velo_track_hit_number_pinned,
  VeloTracking::Hit<true>* host_velo_track_hits_pinned,
  int * host_number_of_tracks_pinned,
  const int &number_of_events
) {

XYZPoint beamspot(0.,0.,0.);
PVSeedTool seedtool;
//std:std::vector<XYZPoint> seeds = seedtool.getSeeds(host_velo_states, beamspot, *host_number_of_tracks_pinned);

/*
AdaptivePV3DFitter fitter;
Vertex recvtx;
std::vector<VeloState> tracks2remove;
XYZPoint seed = seeds.at(0);
fitter.fitVertex(seed, host_velo_states, recvtx, tracks2remove, number_of_events);
*/
std::vector<Vertex> outvtxvec;
std::vector<VeloState> velostate_vec;

for(int i = 0; i < *host_number_of_tracks_pinned; i++)  velostate_vec.push_back(host_velo_states[i]); 
reconstructMultiPVFromTracks(velostate_vec, outvtxvec, *host_number_of_tracks_pinned);


  return 0;
}