#pragma once

#include "cuda_runtime.h"

#include "SystemOfUnits.h"

/**
   *Constants mainly taken from PrVeloUT.h from Rec
   *  @author Mariusz Witek
   *  @date   2007-05-08
   *  @update for A-Team framework 2007-08-20 SHM
   *
   *  2017-03-01: Christoph Hasse (adapt to future framework)
   *  2018-05-05: Plácido Fernández (make standalone)
   *  2018-07:    Dorothea vom Bruch (convert to C, and then to CUDA code)

 */

namespace PrVeloUTConst {
  
  // zMidUT is a position of normalization plane which should
  // to be close to z middle of UT ( +- 5 cm ).
  // No need to update with small UT movement.
  static constexpr float zMidUT = 2484.6;
  //  distToMomentum is properly recalculated in PrUTMagnetTool when B field changes
  static constexpr float distToMomentum = 4.0212e-05;
  static constexpr float sigmaVeloSlope = 0.10*Gaudi::Units::mrad;
  static constexpr float invSigmaVeloSlope = 1./sigmaVeloSlope;
  static constexpr float zKink = 1780.0;

  constexpr float minValsBdl[3] = { -0.3, -250.0, 0.0 };
  constexpr float maxValsBdl[3] = { 0.3, 250.0, 800.0 };
  constexpr float deltaBdl[3]   = { 0.02, 50.0, 80.0 };
  constexpr float dxDyHelper[4] = { 0.0, 1.0, -1.0, 0.0 };
  extern __constant__ float dev_minValsBdl[3];
  extern __constant__ float dev_maxValsBdl[3];
  extern __constant__ float dev_deltaBdl[3];
  extern __constant__ float dev_dxDyHelper[4];
  
  static constexpr float minMomentum =       1.5*Gaudi::Units::GeV;
  static constexpr float minPT =             0.3*Gaudi::Units::GeV;
  static constexpr float maxPseudoChi2 =     1280.;
  static constexpr float yTol =              0.5 * Gaudi::Units::mm;
  static constexpr float yTolSlope =         0.08;
  static constexpr float hitTol1 =           6.0 * Gaudi::Units::mm;
  static constexpr float hitTol2 =           0.8 * Gaudi::Units::mm;
  static constexpr float deltaTx1 =          0.035;
  static constexpr float deltaTx2 =          0.018;
  static constexpr float maxXSlope =         0.350;
  static constexpr float maxYSlope =         0.300;
  static constexpr float centralHoleSize =   33. * Gaudi::Units::mm;
  static constexpr float intraLayerDist =    15.0 * Gaudi::Units::mm;
  static constexpr float overlapTol =        0.7 * Gaudi::Units::mm;
  static constexpr float passHoleSize =      40. * Gaudi::Units::mm;
  static constexpr int   minHighThres =      1;
  static constexpr bool  printVariables =    false;
  static constexpr bool  passTracks =        false;
  static constexpr bool  doTiming =          false;

}
