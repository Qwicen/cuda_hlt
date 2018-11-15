#pragma once

#include <vector>
#include "cuda_runtime.h"

namespace PatPV {


//maximum number of vertices in a event
static constexpr uint max_number_vertices = 30;


// auxiliary class for searching of clusters of tracks


//configuration for seeding
// steering parameters for merging procedure
static constexpr  double mcu_maxChi2Merge = 25.;
static constexpr  double mcu_factorToIncreaseErrors = 15.;

//try parameters from RecoUpgradeTracking.py
static constexpr  int    mcu_minClusterMult = 4;
static constexpr  int    mcu_minCloseTracksInCluster = 3;


// steering parameters for final cluster selection
// int    m_minClusterMult = 3;
static constexpr  double mcu_dzCloseTracksInCluster = 5.; // unit: mm
// int    m_minCloseTracksInCluster = 3;
static constexpr  int    mcu_highMult = 10;
static constexpr  double mcu_ratioSig2HighMult = 1.0;
static constexpr  double mcu_ratioSig2LowMult = 0.9;

static constexpr  int mcu_max_clusters = 200; // maximmum number of clusters

static constexpr  double mcu_x0MS = 0.01;// X0 (tunable) of MS to add for extrapolation of
                                                       // track parameters to PV

//don't forget to actually calculate this!!
//double  m_scatCons = 0;     // calculated from m_x0MS
static constexpr  double X0cu = 0.01;
//static constexpr  double m_scatCons = (13.6*sqrt(X0)*(1.+0.038*log(X0)));
static constexpr  double mcu_scatCons = 0.01;


//configuration for fitting seeds
static constexpr  size_t m_minTr = 4;
static constexpr  int    m_Iterations = 20;
static constexpr  int    m_minIter = 5;
static constexpr  double m_maxDeltaZ = 0.0005; // unit:: mm
static constexpr  double m_minTrackWeight = 0.00000001;
static constexpr  double m_TrackErrorScaleFactor = 1.0;
static constexpr  double m_maxChi2 = 400.0;
static constexpr  double m_trackMaxChi2 = 12.;
//static constexpr  double m_trackChi = std::sqrt(m_trackMaxChi2);     // sqrt of trackMaxChi2
static constexpr  double m_trackChi = 3.464;     // sqrt of trackMaxChi2
static constexpr  double m_trackMaxChi2Remove = 25.;
static constexpr  double m_maxDeltaZCache = 1.; //unit: mm




struct vtxCluster final {

  double  z = 0;            // z of the cluster
  double  sigsq = 0;        // sigma**2 of the cluster
  double  sigsqmin = 0;     // minimum sigma**2 of the tracks forming cluster
  int     ntracks = 1;      // number of tracks in the cluster
  bool     merged = false;   // flag for iterative merging

  vtxCluster() = default;

};

 struct XYZPoint {
  double x = 0.;
  double y = 0.;
  double z = 0.;
   __device__ __host__ XYZPoint(double m_x, double m_y, double m_z) : x(m_x), y(m_y), z(m_z) {};
   __device__ __host__ XYZPoint() {};

};






struct Vector2 {
  double x;
  double y;

  __device__ Vector2(double m_x, double m_y) : x(m_x), y(m_y){}
};

 



class Vertex {
  public:
    __device__ Vertex() {};
    double x = 0.;
    double y = 0.;
    double z = 0.;
    double chi2;
    int ndof;

    double cov00 = 0.;
    double cov10 = 0.;
    double cov11 = 0.;
    double cov20 = 0.;
    double cov21 = 0.;
    double cov22 = 0.;


    __device__ void setChi2AndDoF(double m_chi2, int m_ndof) {
      chi2 = m_chi2;
      ndof = m_ndof;
    }
    __device__ void setPosition(XYZPoint& point) {
      x = point.x;
      y = point.y;
      z = point.z;
    }
    __device__ void setCovMatrix(double * m_cov) {
      cov00 = m_cov[0];
      cov10 = m_cov[1];
      cov11 = m_cov[2];
      cov20 = m_cov[3];
      cov21 = m_cov[4];
      cov22 = m_cov[5];
    }
   int nTracks = 0.;


};

}