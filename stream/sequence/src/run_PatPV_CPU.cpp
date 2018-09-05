#include "run_PatPV_CPU.h"

#include "PVSeedTool.h"




bool reconstructMultiPVFromTracks(VeloState * tracks2use, Vertex * outvtxvec, int host_number_of_tracks_pinned,
  uint * number_of_vertex, int event_number, bool * tracks2disable, XYZPoint * seeds, uint * number_of_seeds) 
{
  

  VeloState * rtracks = tracks2use;
 
  

    
  //PatPv::max_number_vertices


  int nvtx_after  =  0;

    // reconstruct vertices


  
  int number_rec_vtx = 0;
  bool continue_fitting = true;
  while(continue_fitting) {
    XYZPoint beamspot = {0.,0.,0.};
    getSeeds( rtracks, beamspot, host_number_of_tracks_pinned,  seeds, number_of_seeds,  event_number, tracks2disable);
    int before_fit = nvtx_after;
    for(int i=0; i < number_of_seeds[event_number]; i++) {
      XYZPoint seed = seeds[event_number * PatPV::max_number_vertices + i ]; 
      Vertex recvtx;





      // fitting
      bool scvfit = fitVertex( seed, rtracks, recvtx, host_number_of_tracks_pinned, tracks2disable);
      if (!scvfit) continue;

      
      

      
      outvtxvec[event_number * PatPV::max_number_vertices + nvtx_after] = recvtx;
      nvtx_after++;

    }//iterate on seeds
    if(before_fit == nvtx_after) continue_fitting = false;
    
    
  }
    number_of_vertex[event_number] = nvtx_after;


  return true;
  

}




int run_PatPV_on_CPU (
  VeloState * host_velo_states,
  int * host_accumulated_tracks,
  int * host_number_of_tracks_pinned,
  const int &number_of_events,
  Vertex * outvtxvec,
  uint * number_of_vertex,
  XYZPoint * seeds,
  uint * number_of_seeds
) {

XYZPoint beamspot(0.,0.,0.);






for(int i_event = 0; i_event < number_of_events; i_event++) {

  int number_of_tracks = host_number_of_tracks_pinned[i_event];
 VeloState * state_base_pointer = host_velo_states + 2 * host_accumulated_tracks[i_event];
VeloState  kalman_states[number_of_tracks];


 bool  tracks2disable[number_of_tracks];

  //works
for(int i = 0; i < number_of_tracks; i++) kalman_states[i] = state_base_pointer[2*i +1];
for(int i = 0; i < number_of_tracks; i++) tracks2disable[i] = false;

XYZPoint beamspot = {0.,0.,0.};
//getSeeds( kalman_states, beamspot, number_of_tracks,  seeds, number_of_seeds,  i_event, tracks2disable);
reconstructMultiPVFromTracks(kalman_states, outvtxvec, host_number_of_tracks_pinned[i_event], number_of_vertex, i_event, tracks2disable, seeds, number_of_seeds);
}




  return 0;
}


