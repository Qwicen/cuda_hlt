#ifndef PATPV_PVSEEDTOOL_H
#define PATPV_PVSEEDTOOL_H 1




// auxiliary class for searching of clusters of tracks
struct vtxCluster final {

  double  z = 0;            // z of the cluster
  double  sigsq = 0;        // sigma**2 of the cluster
  double  sigsqmin = 0;     // minimum sigma**2 of the tracks forming cluster
  int     ntracks = 0;      // number of tracks in the cluster
  int     not_merged = 0;   // flag for iterative merging

  vtxCluster() = default;

};

struct XYZPoint {
  double x = 0.;
  double y = 0.;
  double z = 0.;
  XYZPoint(double m_x, double m_y, double m_z) : x(m_x), y(m_y), z(m_z) {}

};


struct State {
  double tx = 0.;
  double ty = 0.;
  double x = 0.;
  double y = 0.;
  double z = 0.;
  double errX2 = 0.;
  double errY2 = 0.;


};

//typedef std::vector<State> Track;
class Track {
public:
  std::vector<State> states;
  State firstState() {
    return states.at(0);
  }
  XYZPoint slopes() {
    return XYZPoint(states.at(0).tx, states.at(0).ty, 1.);
  }
  XYZPoint position() {
    return XYZPoint(states.at(0).x, states.at(0).y, states.at(0).z);
  }

};


/** @class PVSeedTool PVSeedTool.h tmp/PVSeedTool.h
 *
 *
 *  @author Mariusz Witek
 *  @date   2005-11-19
 */
class PVSeedTool {
public:

  /// Standard constructor
  PVSeedTool( );

  std::vector<XYZPoint> getSeeds(const std::vector<Track*>& inputTracks,
       const XYZPoint& beamspot) const ;

private:

  std::vector<double> findClusters(std::vector<vtxCluster>& vclus) const;
  void errorForPVSeedFinding(double tx, double ty, double &sigzaq) const;

  double zCloseBeam( Track* track, const XYZPoint& beamspot) const;

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

  double m_x0MS = 0.01;// X0 (tunable) of MS to add for extrapolation of
                                                       // track parameters to PV
  double  m_scatCons = 0;     // calculated from m_x0MS

};
#endif // PATPV_PVSEEDTOOL_H
