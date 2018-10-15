#include "getSeeds.cuh"
//simplficiations: no tracks2disable

// steering parameters for merging procedure
__constant__ double mcu_maxChi2Merge = 25.;
__constant__ double mcu_factorToIncreaseErrors = 15.;

//try parameters from RecoUpgradeTracking.py
__constant__ int    mcu_minClusterMult = 4;
__constant__ int    mcu_minCloseTracksInCluster = 3;


// steering parameters for final cluster selection
// int    m_minClusterMult = 3;
__constant__ double mcu_dzCloseTracksInCluster = 5.; // unit: mm
// int    m_minCloseTracksInCluster = 3;
__constant__ int    mcu_highMult = 10;
__constant__ double mcu_ratioSig2HighMult = 1.0;
__constant__ double mcu_ratioSig2LowMult = 0.9;

__constant__ int mcu_max_clusters = 200; // maximmum number of clusters

__constant__ double mcu_x0MS = 0.01;// X0 (tunable) of MS to add for extrapolation of
                                                       // track parameters to PV

//don't forget to actually calculate this!!
//double  m_scatCons = 0;     // calculated from m_x0MS
__constant__ double X0cu = 0.01;
//__constant__ double m_scatCons = (13.6*sqrt(X0)*(1.+0.038*log(X0)));
__constant__ double mcu_scatCons = 0.01;


__device__ double zCloseBeam( VeloState track, const XYZPoint& beamspot) {

  XYZPoint tpoint(track.x, track.y, track.z);
  XYZPoint tdir(track.tx, track.ty, 1.);

  double wx = ( 1. + tdir.x * tdir.x ) / track.c00;
  double wy = ( 1. + tdir.y * tdir.y ) / track.c11;

  double x0 = tpoint.x - tpoint.z * tdir.x - beamspot.x;
  double y0 = tpoint.y - tpoint.z * tdir.y - beamspot.y;
  double den = wx * tdir.x * tdir.x + wy * tdir.y * tdir.y;
  double zAtBeam = - ( wx * x0 * tdir.x + wy * y0 * tdir.y ) / den ;

  double xb = tpoint.x + tdir.x * ( zAtBeam - tpoint.z ) - beamspot.x;
  double yb = tpoint.y + tdir.y * ( zAtBeam - tpoint.z ) - beamspot.y;
  double r2AtBeam = xb*xb + yb*yb ;

  return r2AtBeam < 0.5*0.5 ? zAtBeam : 10e8;
}



__device__ void errorForPVSeedFinding(double tx, double ty, double &sigz2)  {

    // the seeding results depend weakly on this eror parametrization

    double pMean = 3000.; // unit: MeV

    double tanTheta2 =  tx * tx + ty * ty;
    double sinTheta2 =  tanTheta2 / ( 1. + tanTheta2 );

    // assume that first hit in VD at 8 mm
    double distr        = 8.; // unit: mm
    double dist2        = distr*distr/sinTheta2;
    double sigma_ms2    = mcu_scatCons * mcu_scatCons * dist2 / (pMean*pMean);
    double fslope2      = 0.0005*0.0005;
    double sigma_slope2 = fslope2*dist2;

    sigz2 = (sigma_ms2 + sigma_slope2) / sinTheta2;
    if(sigz2 == 0) sigz2 = 100.;

}




 __global__ void getSeeds(
    VeloState* dev_velo_states,
  int * dev_atomics_storage,
  XYZPoint * dev_seeds,
  uint * dev_number_seed) {

  XYZPoint beamspot;
  beamspot.x = 0;
  beamspot.y = 0;
  beamspot.z = 0;

  int event_number = blockIdx.x;
  int number_of_events = blockDim.x;
   //int * number_of_tracks = dev_atomics_storage;
   //int * acc_tracks = dev_atomics_storage + number_of_events;

  int number_of_tracks = dev_atomics_storage[event_number];

  int acc_tracks = (dev_atomics_storage + number_of_events)[event_number];

  VeloState * state_base_pointer = dev_velo_states + 2 * acc_tracks;

  XYZPoint point;
  point.x = 1.;
  point.y = 1.;
  point.z = 1.;
  int number_seeds = 0;
  for(int i = 0; i < number_of_tracks; i++) {
    point.x = i;
  point.y = i;
  point.z = i;
  number_seeds++;
    dev_seeds[i] = point;
  }

  dev_number_seed[event_number] = number_seeds;

    vtxCluster  vclusters[VeloTracking::max_tracks];



  int counter_number_of_clusters = 0;
  for (int i = 0; i < number_of_tracks; i++) {

    
    double sigsq;
    double zclu;
    auto trk = state_base_pointer[2*i];
   
    zclu = zCloseBeam(trk,beamspot);
    errorForPVSeedFinding(trk.tx, trk.ty,sigsq);

    if ( fabs(zclu)>2000.) continue;
    vtxCluster clu;
    clu.z = zclu;
    clu.sigsq = sigsq;
    clu.sigsqmin = clu.sigsq;
    clu.ntracks = 1;
    vclusters[counter_number_of_clusters] = clu;
    
    counter_number_of_clusters++;

  }


 };