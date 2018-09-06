#include "run_PatPV_CPU.h"

#include "PVSeedTool.h"


//configuration

//separation to accept reconstructed PV
  double m_pvsChi2Separation = 25.;
  double m_pvsChi2SeparationLowMult = 91;
  XYZPoint beamspot = {0.,0.,0.};


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

    getSeeds( rtracks, beamspot, host_number_of_tracks_pinned,  seeds, number_of_seeds,  event_number, tracks2disable);
    int before_fit = nvtx_after;
    for(int i=0; i < number_of_seeds[event_number]; i++) {
      XYZPoint seed = seeds[event_number * PatPV::max_number_vertices + i ]; 
      Vertex recvtx;

      // fitting

      bool tracks2remove[host_number_of_tracks_pinned];

      for(int i = 0; i < host_number_of_tracks_pinned; i++) tracks2remove[i] = false;


      bool scvfit = fitVertex( seed, rtracks, recvtx, host_number_of_tracks_pinned, tracks2disable, tracks2remove);
      if (!scvfit) continue;

      
     

      //only accept vertex if it is not too close too already found one

   

      double chi2min = 1e10;
      for(int i = 0; i < nvtx_after; i++) {
        int index = event_number * PatPV::max_number_vertices + i;
        double z1 = outvtxvec[index].z;
        double z2 = recvtx.z;
        double sigma2z1 = outvtxvec[index].cov22;
        double sigma2z2 = recvtx.cov22;
        double chi2 = pow((z1 - z2), 2) / (sigma2z1 + sigma2z2);
        if (chi2 < chi2min) chi2min = chi2;
      }

      bool vsepar = true;

      if ( chi2min < m_pvsChi2Separation ) vsepar = false;
      // protect secondary vertices of B signal
      //if ( chi2min < m_pvsChi2SeparationLowMult && recvtx.ndof < 7 ) vsepar = false;
      if ( chi2min < m_pvsChi2SeparationLowMult && 0.5*(recvtx.ndof+3) < 7 ) vsepar = false;

      if(!vsepar) continue;

      //remove tracks used

      for(int i = 0; i < host_number_of_tracks_pinned; i++) {
            if(tracks2remove[i]) tracks2disable[i] = true;
      }

      //add reconstructed vertex to output array
      outvtxvec[event_number * PatPV::max_number_vertices + nvtx_after] = recvtx;
          nvtx_after++;

    } //iterate on seeds
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



  for(int i_event = 0; i_event < number_of_events; i_event++) {

    int number_of_tracks = host_number_of_tracks_pinned[i_event];
    VeloState * state_base_pointer = host_velo_states + 2 * host_accumulated_tracks[i_event];
    VeloState  kalman_states[number_of_tracks];


    bool  tracks2disable[number_of_tracks];

    //works
    for(int i = 0; i < number_of_tracks; i++) kalman_states[i] = state_base_pointer[2*i +1];
    for(int i = 0; i < number_of_tracks; i++) tracks2disable[i] = false;

    
    reconstructMultiPVFromTracks(kalman_states, outvtxvec, host_number_of_tracks_pinned[i_event], number_of_vertex, i_event, tracks2disable, seeds, number_of_seeds);
  }

  return 0;
}


