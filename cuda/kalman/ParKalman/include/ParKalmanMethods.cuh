#pragma once

#include "KalmanParametrizations.cuh"
#include "ParKalmanDefinitions.cuh"
#include "ParKalmanMath.cuh"
#include "SciFiConsolidated.cuh"
#include "UTConsolidated.cuh"
#include "VeloConsolidated.cuh"

namespace ParKalmanFilter {

  //----------------------------------------------------------------------
  // Fit information.
  struct trackInfo {

    // Pointer to the extrapolator that should be used.
    // const KalmanParametrizations *m_extr;
    const KalmanParametrizations* m_extr;

    // Jacobians.
    Matrix5x5 m_RefPropForwardTotal;

    // Reference states.
    Vector5 m_RefStateForwardV;

    KalmanFloat m_BestMomEst;
    KalmanFloat m_FirstMomEst;

    // Chi2s.
    KalmanFloat m_chi2;
    KalmanFloat m_chi2T;
    KalmanFloat m_chi2V;

    int m_SciFiLayerIdxs[12];
    int m_UTLayerIdxs[4];

    // NDoFs.
    uint m_ndof;
    uint m_ndofT;
    uint m_ndofUT;
    uint m_ndofV;

    // Number of hits.
    uint m_NHits;
    uint m_NHitsV;
    uint m_NHitsUT;
    uint m_NHitsT;

    // Keep track of the previous UT and T layers.
    uint m_PrevUTLayer;
    uint m_PrevSciFiLayer;

    __device__ __host__ trackInfo()
    {
      for (int i_ut = 0; i_ut < 4; i_ut++)
        m_UTLayerIdxs[i_ut] = -1;
      for (int i_scifi = 0; i_scifi < 12; i_scifi++)
        m_SciFiLayerIdxs[i_scifi] = -1;
    }
  };
} // namespace ParKalmanFilter

using namespace ParKalmanFilter;

////////////////////////////////////////////////////////////////////////
// Functions for doing the extrapolation.
__device__ void
ExtrapolateInV(KalmanFloat zFrom, KalmanFloat zTo, Vector5& x, Matrix5x5& F, SymMatrix5x5& Q, trackInfo& tI);

__device__ bool
ExtrapolateVUT(KalmanFloat zFrom, KalmanFloat zTo, Vector5& x, Matrix5x5& F, SymMatrix5x5& Q, trackInfo& tI);

__device__ void GetNoiseVUTBackw(KalmanFloat zFrom, KalmanFloat zTo, Vector5& x, SymMatrix5x5& Q, trackInfo& tI);

__device__ void ExtrapolateInUT(
  KalmanFloat zFrom,
  uint nLayer,
  KalmanFloat zTo,
  Vector5& x,
  Matrix5x5& F,
  SymMatrix5x5& Q,
  trackInfo& tI);

__device__ void ExtrapolateUTFUTDef(KalmanFloat& zFrom, Vector5& x, Matrix5x5& F, trackInfo& tI);

__device__ void ExtrapolateUTFUT(KalmanFloat zFrom, KalmanFloat zTo, Vector5& x, Matrix5x5& F, trackInfo& tI);

__device__ void ExtrapolateUTT(Vector5& x, Matrix5x5& F, SymMatrix5x5& Q, trackInfo& tI);

__device__ void GetNoiseUTTBackw(const Vector5& x, SymMatrix5x5& Q, trackInfo& tI);

__device__ void ExtrapolateInT(
  KalmanFloat zFrom,
  uint nLayer,
  KalmanFloat& zTo,
  Vector5& x,
  Matrix5x5& F,
  SymMatrix5x5& Q,
  trackInfo& tI);

__device__ void ExtrapolateInT(
  KalmanFloat zFrom,
  uint nLayer,
  KalmanFloat zTo,
  KalmanFloat DzDy,
  KalmanFloat DzDty,
  Vector5& x,
  Matrix5x5& F,
  SymMatrix5x5& Q,
  trackInfo& tI);

__device__ void
ExtrapolateTFT(KalmanFloat zFrom, KalmanFloat& zTo, Vector5& x, Matrix5x5& F, SymMatrix5x5& Q, trackInfo& tI);

__device__ void
ExtrapolateTFTDef(KalmanFloat zFrom, KalmanFloat& zTo, Vector5& x, Matrix5x5& F, SymMatrix5x5& Q, trackInfo& tI);

__device__ int extrapUTT(
  KalmanFloat zi,
  KalmanFloat zf,
  int quad_interp,
  KalmanFloat& x,
  KalmanFloat& y,
  KalmanFloat& tx,
  KalmanFloat& ty,
  KalmanFloat qop,
  KalmanFloat* der_tx,
  KalmanFloat* der_ty,
  KalmanFloat* der_qop,
  trackInfo& tI);

////////////////////////////////////////////////////////////////////////
// Functions for updating and predicting states.
////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------
// Create a seed state at the first VELO hit.
__device__ void CreateVeloSeedState(
  const Velo::Consolidated::Hits& hits,
  const int nVeloHits,
  int nHit,
  Vector5& x,
  SymMatrix5x5& C,
  KalmanFloat& lastz,
  trackInfo& tI);

//----------------------------------------------------------------------
// Predict in the VELO.
__device__ void PredictStateV(
  const Velo::Consolidated::Hits& hits,
  int nHit,
  Vector5& x,
  SymMatrix5x5& C,
  KalmanFloat& lastz,
  trackInfo& tI);

//----------------------------------------------------------------------
// Predict VELO <-> UT.
__device__ bool PredictStateVUT(
  const Velo::Consolidated::Hits& hitsVelo,
  const UT::Consolidated::Hits& hitsUT,
  const int nVeloHits,
  const int nUTHits,
  Vector5& x,
  SymMatrix5x5& C,
  KalmanFloat& lastz,
  trackInfo& tI);

//----------------------------------------------------------------------
// Predict UT <-> UT.
__device__ void PredictStateUT(
  const UT::Consolidated::Hits& hits,
  const uint layer,
  Vector5& x,
  SymMatrix5x5& C,
  KalmanFloat& lastz,
  trackInfo& tI);

//----------------------------------------------------------------------
// Predict last UT layer <-> start of UTTF.
__device__ void PredictStateUTFUT(
  const UT::Consolidated::Hits& hits,
  int nUTHits,
  int forward,
  Vector5& x,
  SymMatrix5x5& C,
  KalmanFloat& lastz,
  trackInfo& tI);

//----------------------------------------------------------------------
// Predict last UT layer <-> start of UTTF.
__device__ void PredictStateUTFUT(int forward, Vector5& x, SymMatrix5x5& C, KalmanFloat& lastz, trackInfo& tI);

//----------------------------------------------------------------------
// Predict UT <-> T precise version(?)
__device__ void PredictStateUTT(
  const UT::Consolidated::Hits& hits,
  const int n_ut_hits,
  Vector5& x,
  SymMatrix5x5& C,
  KalmanFloat& lastz,
  trackInfo& tI);

//----------------------------------------------------------------------
// Predict T <-> T.
__device__ void PredictStateT(
  const SciFi::Consolidated::Hits& hits,
  uint layer,
  Vector5& x,
  SymMatrix5x5& C,
  KalmanFloat& lastz,
  trackInfo& tI);

//----------------------------------------------------------------------
// Predict T(fixed z=7783) <-> first T layer.
__device__ void PredictStateTFT(
  const SciFi::Consolidated::Hits& hits,
  int forward,
  Vector5& x,
  SymMatrix5x5& C,
  KalmanFloat& lastz,
  trackInfo& tI);

//----------------------------------------------------------------------
// Predict T(fixed z=7783) <-> first T layer.
__device__ void PredictStateTFT(int forward, Vector5& x, SymMatrix5x5& C, KalmanFloat& lastz, trackInfo& tI);

//----------------------------------------------------------------------
// Update state with velo measurement.
__device__ void
UpdateStateV(const Velo::Consolidated::Hits& hits, int forward, int nHit, Vector5& x, SymMatrix5x5& C, trackInfo& tI);

//----------------------------------------------------------------------
// Update state with UT measurement.
__device__ void UpdateStateUT(
  const UT::Consolidated::Hits& hits,
  uint layer,
  Vector5& x,
  SymMatrix5x5& C,
  KalmanFloat& lastz,
  trackInfo& tI);

//----------------------------------------------------------------------
// Update state with T measurement.
__device__ void UpdateStateT(
  const SciFi::Consolidated::Hits& hits,
  int forward,
  uint layer,
  Vector5& x,
  SymMatrix5x5& C,
  KalmanFloat& lastz,
  trackInfo& tI);

//----------------------------------------------------------------------
// Extrapolate to the vertex using straight line extrapolation.
__device__ void ExtrapolateToVertex(Vector5& x, SymMatrix5x5& C, KalmanFloat& lastz);
