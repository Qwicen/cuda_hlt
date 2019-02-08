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
  static constexpr float zMidUT = 2484.6f;
  //  distToMomentum is properly recalculated in PrUTMagnetTool when B field changes
  static constexpr float distToMomentum = 4.0212e-05f;
  static constexpr float sigmaVeloSlope = 0.10f * Gaudi::Units::mrad;
  static constexpr float invSigmaVeloSlope = 1.0f / sigmaVeloSlope;
  static constexpr float zKink = 1780.0f;

  static constexpr float minMomentum = 1.5f * Gaudi::Units::GeV;
  static constexpr float minPT = 0.3f * Gaudi::Units::GeV;
  static constexpr float maxPseudoChi2 = 1280.0f;
  static constexpr float yTol = 0.5f * Gaudi::Units::mm;
  static constexpr float yTolSlope = 0.08f;
  static constexpr float hitTol1 = 6.0f * Gaudi::Units::mm;
  static constexpr float hitTol2 = 0.8f * Gaudi::Units::mm;
  static constexpr float deltaTx1 = 0.035f;
  static constexpr float deltaTx2 = 0.018f;
  static constexpr float maxXSlope = 0.350f;
  static constexpr float maxYSlope = 0.300f;
  static constexpr float centralHoleSize = 33.0f * Gaudi::Units::mm;
  static constexpr float intraLayerDist = 15.0f * Gaudi::Units::mm;
  static constexpr float overlapTol = 0.7f * Gaudi::Units::mm;
  static constexpr float passHoleSize = 40.0f * Gaudi::Units::mm;
  static constexpr int minHighThres = 1;
  static constexpr bool printVariables = false;
  static constexpr bool passTracks = false;
  static constexpr bool doTiming = false;
  // Scale the z-component, to not run into numerical problems with floats
  // first add to sum values from hit at xMidField, zMidField hit
  static constexpr float zDiff = 0.001f * (zKink - zMidUT);

} // namespace PrVeloUTConst
