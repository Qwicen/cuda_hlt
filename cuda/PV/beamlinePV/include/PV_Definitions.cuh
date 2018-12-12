#pragma once

#include "cuda_runtime.h"

namespace PV {

typedef float myfloat;

// Number of threads pv_beamline_peak_t
static constexpr uint num_threads_pv_beamline_peak_t = 64;

// maximum number of vertices in a event
static constexpr uint max_number_vertices = 32;

// study this
static constexpr uint max_number_subclusters = 50;

// STUDY THIS NUMBER
static constexpr uint max_number_of_clusters = 200;

// STUDY THIS
static constexpr uint max_number_clusteredges = 200;
// auxiliary class for searching of clusters of tracks

// configuration for seeding
// steering parameters for merging procedure
static constexpr myfloat mcu_maxChi2Merge = 25.f;
static constexpr myfloat mcu_factorToIncreaseErrors = 15.f;

// try parameters from RecoUpgradeTracking.py
static constexpr int mcu_minClusterMult = 4;
static constexpr int mcu_minCloseTracksInCluster = 3;

// steering parameters for final cluster selection
// int    m_minClusterMult = 3;
static constexpr myfloat mcu_dzCloseTracksInCluster = 5.f; // unit: mm
// int    m_minCloseTracksInCluster = 3;
static constexpr int mcu_highMult = 10;
static constexpr myfloat mcu_ratioSig2HighMult = 1.0f;
static constexpr myfloat mcu_ratioSig2LowMult = 0.9f;

static constexpr int mcu_max_clusters = 1200; // maximmum number of clusters

static constexpr myfloat mcu_x0MS = 0.01f; // X0 (tunable) of MS to add for extrapolation of
                                          // track parameters to PV

// don't forget to actually calculate this!!
// double  m_scatCons = 0;     // calculated from m_x0MS
static constexpr myfloat X0cu = 0.01f;
// static constexpr  double m_scatCons = (13.6*sqrt(X0)*(1.+0.038*log(X0)));
static constexpr myfloat mcu_scatCons = 0.01f;

// configuration for fitting seeds
static constexpr size_t m_minTr = 4;
static constexpr int m_Iterations = 20;
static constexpr int m_minIter = 5;
static constexpr myfloat m_maxDeltaZ = 0.0005f; // unit:: mm
static constexpr myfloat m_minTrackWeight = 0.00000001f;
static constexpr myfloat m_TrackErrorScaleFactor = 1.0f;
static constexpr myfloat m_maxChi2 = 400.0f;
static constexpr myfloat m_trackMaxChi2 = 12.f;
// static constexpr  double m_trackChi = std::sqrt(m_trackMaxChi2);     // sqrt of trackMaxChi2
static constexpr myfloat m_trackChi = 3.464f; // sqrt of trackMaxChi2
static constexpr myfloat m_trackMaxChi2Remove = 25.f;
static constexpr myfloat m_maxDeltaZCache = 1.f; // unit: mm

struct vtxCluster final {

  myfloat z = 0.f;        // z of the cluster
  myfloat sigsq = 0.f;    // sigma**2 of the cluster
  myfloat sigsqmin = 0.f; // minimum sigma**2 of the tracks forming cluster
  int ntracks = 1;      // number of tracks in the cluster
  bool merged = false;  // flag for iterative merging

  vtxCluster() = default;
};

class Vertex {
public:
  __host__ __device__ Vertex() {};
  float3 position;
  myfloat chi2;
  int ndof;
  uint n_tracks = 0.f;

  myfloat cov00 = 0.f;
  myfloat cov10 = 0.f;
  myfloat cov11 = 0.f;
  myfloat cov20 = 0.f;
  myfloat cov21 = 0.f;
  myfloat cov22 = 0.f;

  __host__ __device__ void setChi2AndDoF(myfloat m_chi2, int m_ndof)
  {
    chi2 = m_chi2;
    ndof = m_ndof;
  }
  __host__ __device__ void setPosition(float3& point) { position = point; }
  __host__ __device__ void setCovMatrix(myfloat* m_cov)
  {
    cov00 = m_cov[0];
    cov10 = m_cov[1];
    cov11 = m_cov[2];
    cov20 = m_cov[3];
    cov21 = m_cov[4];
    cov22 = m_cov[5];
  }
  int nTracks = 0;
};

} // namespace PV
