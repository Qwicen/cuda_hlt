#ifndef PATPV_PVSEEDTOOL_H
#define PATPV_PVSEEDTOOL_H 1



#include "../../cuda/velo/common/include/VeloDefinitions.cuh"
#include "../../cuda/patPV/include/patPV_Definitions.cuh"
using namespace PatPV ;

/** @class PVSeedTool PVSeedTool.h tmp/PVSeedTool.h
 *
 *
 *  @author Mariusz Witek
 *  @date   2005-11-19
 */
class PVSeedTool {
public:

  /// Standard constructor
  PVSeedTool( ) {};

  std::vector<XYZPoint> getSeeds( VeloState* inputTracks,
       const XYZPoint& beamspot, int number_of_tracks)  ;

   void resetClusterCounter()  {m_cluster_counter = 0 ;}
   void increaseClusterCounter()  {m_cluster_counter++ ;}
   int getClusterCounter()  {return m_cluster_counter;}

private:

  void findClusters(vtxCluster * vclus, double * zclusters, int number_of_clusters) ;
  void errorForPVSeedFinding(double tx, double ty, double &sigzaq) const;

  double zCloseBeam(  VeloState track, const XYZPoint& beamspot) const;

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

  int m_max_clusters = 200; // maximmum nubmer of clusters 
  int m_cluster_counter; //counts how many cluster we have
  double m_x0MS = 0.01;// X0 (tunable) of MS to add for extrapolation of
                                                       // track parameters to PV
  double  m_scatCons = 0;     // calculated from m_x0MS

};
#endif // PATPV_PVSEEDTOOL_H
