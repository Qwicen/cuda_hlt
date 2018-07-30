#ifndef PATPV_PVSEEDTOOL_H
#define PATPV_PVSEEDTOOL_H 



#include "../../cuda/velo/common/include/VeloDefinitions.cuh"
#include "../../cuda/patPV/include/patPV_Definitions.cuh"


/** @class PVSeedTool PVSeedTool.h tmp/PVSeedTool.h
 *
 *
 *  @author Mariusz Witek
 *  @date   2005-11-19
 */



  void getSeeds( VeloState* inputTracks,
       const XYZPoint& beamspot, int number_of_tracks, XYZPoint * seeds, uint * number_of_seeds,  int event_number)  ;




  int findClusters(vtxCluster * vclus, double * zclusters, int number_of_clusters) ;
  void errorForPVSeedFinding(double tx, double ty, double &sigzaq) ;

  double zCloseBeam(  VeloState track, const XYZPoint& beamspot) ;

  // steering parameters for merging procedure
  double m_maxChi2Merge = 25.;
  double m_factorToIncreaseErrors = 15.;

  // steering parameters for final cluster selection
  int    m_minClusterMult = 3;
  double m_dzCloseTracksInCluster = 5.; // unit: mm
  int    m_minCloseTracksInCluster = 3;
  int    m_highMult = 10;
  double m_ratioSig2HighMult = 1.0;
  double m_ratioSig2LowMult = 0.9;

  int m_max_clusters = 200; // maximmum number of clusters 

  double m_x0MS = 0.01;// X0 (tunable) of MS to add for extrapolation of
                                                       // track parameters to PV
  double  m_scatCons = 0;     // calculated from m_x0MS


#endif PATPV_PVSEEDTOOL_H // PATPV_PVSEEDTOOL_H
