#include "patPV.cuh"


__device__ void reconstructMultiPVFromTracks(Velo::State * tracks2use, Vertex * outvtxvec, int number_of_tracks,
  uint * number_of_vertex,  bool * tracks2disable) 
{
  /*
  const int maxnumberseeds = 200;
  XYZPoint seeds[maxnumberseeds];
  uint number_of_seeds[20];

  VeloState * rtracks = tracks2use;
  XYZPoint beamspot;
  getSeeds( rtracks, beamspot, seeds, number_of_seeds, tracks2disable);

  // reconstruct vertices

  int nvtx_after  =  0;
  bool continue_fitting = true;
 
 while(continue_fitting) {

    getSeeds( rtracks, beamspot, seeds, number_of_seeds, tracks2disable);
    int before_fit = nvtx_after;
    for(int i=0; i < number_of_seeds[event_number]; i++) {
      XYZPoint seed = seeds[event_number * PatPV::max_number_vertices + i ]; 
      Vertex recvtx;

      // fitting

      bool tracks2remove[number_of_tracks];

      for(int i = 0; i < number_of_tracks; i++) tracks2remove[i] = false;


      bool scvfit = fitVertex( seed, rtracks, recvtx, number_of_tracks, tracks2disable, tracks2remove);
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

      for(int i = 0; i < number_of_tracks; i++) {
            if(tracks2remove[i]) tracks2disable[i] = true;
      }

      //add reconstructed vertex to output array
      outvtxvec[event_number * PatPV::max_number_vertices + nvtx_after] = recvtx;
      nvtx_after++;

    } //iterate on seeds

    if(before_fit == nvtx_after) continue_fitting = false;
    
    
  }
    number_of_vertex[event_number] = nvtx_after;


 */
  

}


__global__ void patPV(
  Velo::State* dev_velo_states,
  int * dev_atomics_storage,
  Vertex * dev_outvtxvec,
  uint * dev_number_of_vertex
) {

   int number_of_events = blockDim.x;
   //int * number_of_tracks = dev_atomics_storage;
   //int * acc_tracks = dev_atomics_storage + number_of_events;

   int number_of_tracks = dev_atomics_storage[number_of_events];
   int acc_tracks = (dev_atomics_storage + number_of_events)[number_of_events];

   Velo::State * state_base_pointer = dev_velo_states + 2 * acc_tracks;

  const int maxnumbertracks = 200;
    //vector with states from Kalman fit
  Velo::State  kalman_states[maxnumbertracks];


  bool  tracks2disable[maxnumbertracks];

    //stride because we saved tracks from straight line fit and Kalman fit in same vector
    for(int i = 0; i < number_of_tracks; i++) kalman_states[i] = state_base_pointer[2*i +1];
    for(int i = 0; i < number_of_tracks; i++) tracks2disable[i] = false;

   reconstructMultiPVFromTracks(kalman_states, dev_outvtxvec, number_of_tracks, dev_number_of_vertex, tracks2disable);
 



};
