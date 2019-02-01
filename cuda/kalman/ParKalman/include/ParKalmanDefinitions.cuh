#pragma once

#include "ParKalmanMath.cuh"

namespace ParKalmanFilter {
  
  typedef Vector<5> Vector5;
  typedef SquareMatrix<true,5> SymMatrix5x5;
  typedef SquareMatrix<false,5> Matrix5x5;

    // Set a 5x5 diagonal matrix for later use
  __constant__ static double F_diag[25] = {1,0,0,0,0,
                                           0,1,0,0,0,
                                           0,0,1,0,0,
                                           0,0,0,1,0,
                                           0,0,0,0,1};
  
  // Max number of measurements.
  const int nMaxMeasurements = 41; // 25 VELO + 4 UT + 12 SciFi

  // Max number of bins for the UT <-> SciFi extrapolation.
  const int nBinXMax = 60;
  const int nBinYMax = 50;
  
  // Number of velo parameters.
  const int nParsV = 10;
  const int nSetsV = 2;

  // Number of velo-UT parameters.
  const int nParsVUT = 30;
  const int nSetsVUT = 2;

  // Number of UT parameters.
  const int nParsUT = 20;
  const int nSetsUT = 7;

  // Number of UTFUT parameters.
  const int nParsUTFUT = 1;
  const int nSetsUTFUT = 1;

  // Number of UTTF parameters.
  const int nParsUTTF = 20;
  const int nSetsUTTF = 2;
  
  // Number of TFT parameters.
  const int nParsTFT = 20;
  const int nSetsTFT = 2;

  // Number of T parameters.
  const int nParsT = 20;
  const int nSetsT = 46;

  // Number of TLayer parameters.
  const int nParsTLayer = 12;
  const int nSetsTLayer = 2;

  // Number of UTLayer parameters.
  const int nParsUTLayer = 4;
  const int nSetsUTLayer = 1;

  // Some options.
  const bool m_UseForwardMomEstimate = true;
  const bool m_UseForwardChi2Estimate = true;
  const int nMaxOutliers = 2;

  //----------------------------------------------------------------------
  // Tentative output structure.
  struct FittedTrack {    
    // Chi2 info.
    double chi2;
    double chi2V;
    double chi2T;
    
    // Ns DoF
    uint ndof;
    uint ndofV;
    uint ndofT;
    uint nhits;
    
    // State info (closest to beamline).
    Vector5 state;
    SymMatrix5x5 cov;    
    double z;
  };
  
}
