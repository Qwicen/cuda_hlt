#pragma once

#include "ParKalmanMath.cuh"
#include "ParKalmanDefinitions.cuh"
#include "KalmanParametrizations.cuh"
#include "VeloConsolidated.cuh"
#include "UTConsolidated.cuh"
#include "SciFiConsolidated.cuh"

namespace ParKalmanFilter {
  
  //----------------------------------------------------------------------
  // Fit information.
  struct trackInfo {
    
    // Pointer to the extrapolator that should be used.
    const KalmanParametrizations *m_extr;
        
    // Jacobians.
    Matrix5x5 m_RefPropForwardVUT;
    Matrix5x5 m_RefPropForwardUTT;

    // Reference states.
    Vector5 m_RefStateForwardV;
    Vector5 m_RefStateForwardFUT;
    Vector5 m_RefStateForwardUT;
    Vector5 m_RefStateForwardT;

    double m_StateZPos[nMaxMeasurements];
    double m_HitChi2[nMaxMeasurements];

    double m_BestMomEst;
    
    // Chi2s.
    double m_chi2;
    double m_chi2T;
    double m_chi2V;

    // Hit mask (allows for removing hits).
    int m_HitStatus[nMaxMeasurements];

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

    int m_charge;

    // Keep track of the previous UT and T layers.
    uint m_PrevUTLayer;
    uint m_PrevSciFiLayer;

    // Options.
    bool m_do_smoother;

    __device__ __host__ trackInfo(){
      for(int i_ut=0; i_ut<4; i_ut++)
        m_UTLayerIdxs[i_ut] = -1;
      for(int i_scifi=0; i_scifi<12; i_scifi++)
        m_SciFiLayerIdxs[i_scifi] = -1;
    }
    
  };
}

using namespace ParKalmanFilter;

////////////////////////////////////////////////////////////////////////
// Functions for doing the extrapolation.
__device__ void ExtrapolateInV(
  double zFrom,
  double zTo,
  Vector5 &x,
  Matrix5x5 &F,
  SymMatrix5x5 &Q,
  trackInfo &tI
);

__device__ bool ExtrapolateVUT(
  double zFrom,
  double zTo,
  Vector5 &x,
  Matrix5x5 &F,
  SymMatrix5x5 &Q,
  trackInfo &tI
);

__device__ void GetNoiseVUTBackw(
  double zFrom,
  double zTo,
  Vector5 &x,
  SymMatrix5x5 &Q,
  trackInfo &tI
);

__device__ void ExtrapolateInUT(
  double zFrom,
  uint nLayer,
  double zTo,
  Vector5 &x,
  Matrix5x5 &F,
  SymMatrix5x5 &Q,
  trackInfo &tI
);

__device__ void ExtrapolateUTFUTDef(
  double &zFrom,
  Vector5 &x,
  Matrix5x5 &F,
  trackInfo &tI
);

__device__ void ExtrapolateUTFUT(
  double zFrom,
  double zTo,
  Vector5 &x,
  Matrix5x5 &F,
  trackInfo &tI
);

__device__ void ExtrapolateUTT(
  Vector5 &x,
  Matrix5x5 &F,
  SymMatrix5x5 &Q,
  trackInfo &tI
);

__device__ void GetNoiseUTTBackw(
  const Vector5 &x,
  SymMatrix5x5 &Q,
  trackInfo &tI
);

__device__ void ExtrapolateInT(
  double zFrom,
  uint nLayer,
  double &zTo,
  Vector5 &x,
  Matrix5x5 &F,
  SymMatrix5x5 &Q,
  trackInfo &tI
);

__device__ void ExtrapolateInT(
  double zFrom,
  uint nLayer,
  double zTo,
  double DzDy,
  double DzDty,
  Vector5 &x,
  Matrix5x5 &F,
  SymMatrix5x5 &Q,
  trackInfo &tI
);

__device__ void ExtrapolateTFT(
  double zFrom,
  double &zTo,
  Vector5 &x,
  Matrix5x5 &F,
  SymMatrix5x5 &Q,
  trackInfo &tI
);

__device__ void ExtrapolateTFTDef(
  double zFrom,
  double &zTo,
  Vector5 &x,
  Matrix5x5 &F,
  SymMatrix5x5 &Q,
  trackInfo &tI
);

__device__ int extrapUTT(
  double zi,
  double zf,
  int quad_interp,
  double& x,
  double& y,
  double& tx,
  double& ty,
  double qop,
  double* der_tx,
  double* der_ty,
  double* der_qop,
  trackInfo &tI
);

  ////////////////////////////////////////////////////////////////////////
  // Functions for updating and predicting states.
  ////////////////////////////////////////////////////////////////////////

  //----------------------------------------------------------------------
  // Create a seed state at the first VELO hit.
  __device__ void CreateVeloSeedState(
    const Velo::Consolidated::Hits &hits,
    const int nVeloHits,
    int nHit,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  );

  /* TODO: Decide how to implement this general update state method.
  //----------------------------------------------------------------------
  // Update the state at a hit.
  __device__ __host__ void UpdateState(
    const Velo::Consolidated::Hits &hits,
    int forward,
    int nHit,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  );
  */
  
  //----------------------------------------------------------------------
  // Predict in the VELO.
  __device__ void PredictStateV(
    const Velo::Consolidated::Hits &hits,
    int nHit,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  );

  //----------------------------------------------------------------------
  // Predict VELO <-> UT.
  __device__ bool PredictStateVUT(
    const Velo::Consolidated::Hits &hitsVelo,                             
    const UT::Consolidated::Hits &hitsUT,
    const int nVeloHits,
    const int nUTHits,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  );

  //----------------------------------------------------------------------
  // Predict UT <-> UT.
  __device__ void PredictStateUT(
    const UT::Consolidated::Hits &hits,
    const uint layer,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  );

  //----------------------------------------------------------------------
  // Predict last UT layer <-> start of UTTF.
  __device__ void PredictStateUTFUT(
    const UT::Consolidated::Hits &hits,
    int nUTHits,
    int forward,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  );

  //----------------------------------------------------------------------
  // Predict last UT layer <-> start of UTTF.
  __device__ void PredictStateUTFUT(
    int forward,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  );

  //----------------------------------------------------------------------
  // Predict UT <-> T precise version(?)
  __device__ void PredictStateUTT(
    const UT::Consolidated::Hits &hits,
    const int n_ut_hits,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  );
  
  //----------------------------------------------------------------------
  // Predict T <-> T.
  __device__ void PredictStateT(
    const SciFi::Consolidated::Hits &hits,
    uint layer,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  );

  //----------------------------------------------------------------------
  // Predict T(fixed z=7783) <-> first T layer.
  __device__ void PredictStateTFT(
    const SciFi::Consolidated::Hits &hits,
    int forward,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  );

  //----------------------------------------------------------------------
  // Predict T(fixed z=7783) <-> first T layer.
  __device__ void PredictStateTFT(
    int forward,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  );

  //----------------------------------------------------------------------
  // Update state with velo measurement.
  __device__ void UpdateStateV(
    const Velo::Consolidated::Hits &hits,
    int forward,
    int nHit,
    Vector5 &x,
    SymMatrix5x5 &C,
    trackInfo &tI
  );

  //----------------------------------------------------------------------
  // Update state with UT measurement.
  __device__ void UpdateStateUT(
    const UT::Consolidated::Hits &hits,
    uint layer,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  );

  //----------------------------------------------------------------------
  // Update state with T measurement.
  __device__ void UpdateStateT(
    const SciFi::Consolidated::Hits &hits,
    int forward,
    uint layer,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  );

  //----------------------------------------------------------------------
  // Smoothe/average.
  template<typename HitType>
  __device__ bool AverageState(
    const HitType &hits,
    int nHit,
    trackInfo &tI
  );

  //----------------------------------------------------------------------
  // Extrapolate to the vertex using straight line extrapolation.
  __device__ void ExtrapolateToVertex(
    Vector5 &x, SymMatrix5x5 &C, double &lastz
  );

  //----------------------------------------------------------------------
  // Create the track.
  //
  // TODO: Figure out what to actually do here.
  
  //----------------------------------------------------------------------
  // Check if outliers should be removed and remove one of them.
  __device__ bool DoOutlierRemoval(trackInfo &tI);
  
