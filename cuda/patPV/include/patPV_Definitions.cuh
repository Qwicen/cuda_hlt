#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <vector>
#include "../../velo/common/include/VeloDefinitions.cuh"





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






struct Vector2 {
  double x;
  double y;

  Vector2(double m_x, double m_y) : x(m_x), y(m_y){}
};

 



class Vertex {
  public:
    Vertex() {};
    XYZPoint pos{0.,0.,0.};
    double chi2;
    int ndof;
    double cov[6];
    std::vector<VeloState> tracks;
    std::vector<double> weights;
    void setChi2AndDoF(double m_chi2, int m_ndof) {
      chi2 = m_chi2;
      ndof = m_ndof;
    }
    void setPosition(XYZPoint& point) {
      pos.x = point.x;
      pos.y = point.y;
      pos.z = point.z;
    }
    void setCovMatrix(double * m_cov) {
      cov[0] = m_cov[0];
      cov[1] = m_cov[1];
      cov[2] = m_cov[2];
      cov[3] = m_cov[3];
      cov[4] = m_cov[4];
      cov[5] = m_cov[5];
    }

    void clearTracks() {
      tracks.clear();
      weights.clear();
    };
    void addToTracks(VeloState track, double weight) {
      tracks.push_back(track);
      weights.push_back(weight);
    };
};

#endif DEFINITIONS_H