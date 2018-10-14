#pragma once

#include "SciFiDefinitions.cuh"
#include "TrackUtils.cuh"
#include "HitUtils.cuh"

#include <cmath>

namespace SciFi {
  namespace Tracking {

    struct LineFitterPars {
      float   m_z0 = 0.; 
      float   m_c0 = 0.; 
      float   m_tc = 0.; 
      
      float m_s0 = 0.; 
      float m_sz = 0.; 
      float m_sz2 = 0.; 
      float m_sc = 0.; 
      float m_scz = 0.;   
    };
  }
}

__host__ __device__ void incrementLineFitParameters(
  SciFi::Tracking::LineFitterPars &parameters,
  const SciFi::SciFiHits& scifi_hits,
  const float coordX[SciFi::Tracking::max_x_hits],
  const int allXHits[SciFi::Tracking::max_x_hits],
  const int it);

__host__ __device__ void fitHitsFromSingleHitPlanes(
  const int it1,
  const int it2,
  const bool usedHits[SciFi::Tracking::max_x_hits],
  const SciFi::SciFiHits& scifi_hits,
  const int allXHits[SciFi::Tracking::max_x_hits],
   const int n_x_hits,
  const PlaneCounter planeCounter,
  SciFi::Tracking::LineFitterPars& lineFitParameters,
  const float coordX[SciFi::Tracking::max_x_hits],
  int otherHits[SciFi::Constants::n_layers][SciFi::Tracking::max_other_hits],
  int nOtherHits[SciFi::Constants::n_layers] );

__host__ __device__ void addAndFitHitsFromMultipleHitPlanes(
  const int nOtherHits[SciFi::Constants::n_layers],
  SciFi::Tracking::LineFitterPars& lineFitParameters,
  const SciFi::SciFiHits& scifi_hits,
  const float coordX[SciFi::Tracking::max_x_hits],
  const int allXHits[SciFi::Tracking::max_x_hits],
  const int otherHits[SciFi::Constants::n_layers][SciFi::Tracking::max_other_hits]);

__host__ __device__ float getLineFitDistance(
  SciFi::Tracking::LineFitterPars &parameters,
  const SciFi::SciFiHits& scifi_hits,
  float coordX[SciFi::Tracking::max_x_hits],
  int allXHits[SciFi::Tracking::max_x_hits],
  int it );

__host__ __device__ float getLineFitChi2(
  SciFi::Tracking::LineFitterPars &parameters,
  const SciFi::SciFiHits& scifi_hits,
  float coordX[SciFi::Tracking::max_x_hits],
  int allXHits[SciFi::Tracking::max_x_hits],
  int it);

__host__ __device__ void solveLineFit(SciFi::Tracking::LineFitterPars &parameters);

__host__ __device__ void fastLinearFit(
  const SciFi::SciFiHits& scifi_hits,
  float trackParameters[SciFi::Tracking::nTrackParams],
  int coordToFit[SciFi::Tracking::max_coordToFit],
  int& n_coordToFit,
  PlaneCounter planeCounter,
  SciFi::Tracking::HitSearchCuts& pars_cur);
