#pragma once

#include "cuda_runtime.h"

namespace PV {

typedef float myfloat;


//maximum number of vertices in a event
static constexpr uint max_number_vertices = 30;


// auxiliary class for searching of clusters of tracks


//configuration for seeding
// steering parameters for merging procedure
static constexpr  myfloat mcu_maxChi2Merge = 25.;
static constexpr  myfloat mcu_factorToIncreaseErrors = 15.;

//try parameters from RecoUpgradeTracking.py
static constexpr  int    mcu_minClusterMult = 4;
static constexpr  int    mcu_minCloseTracksInCluster = 3;


// steering parameters for final cluster selection
// int    m_minClusterMult = 3;
static constexpr  myfloat mcu_dzCloseTracksInCluster = 5.; // unit: mm
// int    m_minCloseTracksInCluster = 3;
static constexpr  int    mcu_highMult = 10;
static constexpr  myfloat mcu_ratioSig2HighMult = 1.0;
static constexpr  myfloat mcu_ratioSig2LowMult = 0.9;

static constexpr  int mcu_max_clusters = 1200; // maximmum number of clusters

static constexpr  myfloat mcu_x0MS = 0.01;// X0 (tunable) of MS to add for extrapolation of
                                                       // track parameters to PV

//don't forget to actually calculate this!!
//double  m_scatCons = 0;     // calculated from m_x0MS
static constexpr  myfloat X0cu = 0.01;
//static constexpr  double m_scatCons = (13.6*sqrt(X0)*(1.+0.038*log(X0)));
static constexpr  myfloat mcu_scatCons = 0.01;


//configuration for fitting seeds
static constexpr  size_t m_minTr = 4;
static constexpr  int    m_Iterations = 20;
static constexpr  int    m_minIter = 5;
static constexpr  myfloat m_maxDeltaZ = 0.0005; // unit:: mm
static constexpr  myfloat m_minTrackWeight = 0.00000001;
static constexpr  myfloat m_TrackErrorScaleFactor = 1.0;
static constexpr  myfloat m_maxChi2 = 400.0;
static constexpr  myfloat m_trackMaxChi2 = 12.;
//static constexpr  double m_trackChi = std::sqrt(m_trackMaxChi2);     // sqrt of trackMaxChi2
static constexpr  myfloat m_trackChi = 3.464;     // sqrt of trackMaxChi2
static constexpr  myfloat m_trackMaxChi2Remove = 25.;
static constexpr  myfloat m_maxDeltaZCache = 1.; //unit: mm


struct vtxCluster final {

  myfloat  z = 0;            // z of the cluster
  myfloat  sigsq = 0;        // sigma**2 of the cluster
  myfloat  sigsqmin = 0;     // minimum sigma**2 of the tracks forming cluster
  int     ntracks = 1;      // number of tracks in the cluster
  bool     merged = false;   // flag for iterative merging

  vtxCluster() = default;

};

 struct XYZPoint {
  myfloat x = 0.;
  myfloat y = 0.;
  myfloat z = 0.;
   __device__ __host__ XYZPoint(myfloat m_x, myfloat m_y, myfloat m_z) : x(m_x), y(m_y), z(m_z) {};
   __device__ __host__ XYZPoint() {};

};

struct Vector2 {
  myfloat x;
  myfloat y;

  __device__ __host__ Vector2(myfloat m_x, myfloat m_y) : x(m_x), y(m_y){}
};
  
  class Vertex {
  public:
    __device__ Vertex() {};
    float3 position;
    myfloat chi2;
    int ndof;
    std::vector<std::pair<unsigned,float> > tracks ;

    myfloat cov00 = 0.;
    myfloat cov10 = 0.;
    myfloat cov11 = 0.;
    myfloat cov20 = 0.;
    myfloat cov21 = 0.;
    myfloat cov22 = 0.;


    __device__ void setChi2AndDoF(myfloat m_chi2, int m_ndof) {
      chi2 = m_chi2;
      ndof = m_ndof;
    }
    __device__ void setPosition(float3& point) {
      position = point;
    }
    __device__ void setCovMatrix(myfloat * m_cov) {
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
