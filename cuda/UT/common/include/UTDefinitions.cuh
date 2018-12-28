#pragma once

#include "SystemOfUnits.h"
#include "VeloEventModel.cuh"
#include "VeloConsolidated.cuh"
#include "UTRaw.cuh"

#include <ostream>
#include <stdint.h>
#include <vector>

namespace UT {
namespace Decoding {

static constexpr int frac_mask = 0x0003U; // frac
static constexpr int chan_mask = 0x3FFCU; // channel
static constexpr int thre_mask = 0x8000U; // threshold

static constexpr int frac_offset = 0; // frac
static constexpr int chan_offset = 2; // channel
static constexpr int thre_offset = 15; // threshold

static constexpr uint ut_number_of_sectors_per_board = 6;
static constexpr uint ut_number_of_geometry_sectors = 1048;
static constexpr uint ut_decoding_in_order_threads_x = 64;
static constexpr uint ut_max_hits_shared_sector_group = 256;

} // namespace Decoding

static constexpr int num_atomics = 3;

namespace Constants {
/* Detector description
   There are two stations with two layers each
*/
static constexpr uint n_layers = 4;
static constexpr uint n_regions_in_layer = 3;

/* Cut-offs */
static constexpr uint max_numhits_per_layer = 500;
static constexpr uint max_numhits_per_event = 6000;
static constexpr uint max_hit_candidates_per_layer = 100;
static constexpr uint max_num_tracks = 400; // to do: what is the best / safest value here?
static constexpr uint max_track_size = 4;

// zMidUT is a position of normalization plane which should
// to be close to z middle of UT ( +- 5 cm ).
// No need to update with small UT movement.
static constexpr float zMidUT = 2484.6f;
//  distToMomentum is properly recalculated in PrUTMagnetTool when B field changes
static constexpr float distToMomentum = 4.0212e-05f;
static constexpr float sigmaVeloSlope = 0.10f*Gaudi::Units::mrad;
static constexpr float invSigmaVeloSlope = 1.0f/sigmaVeloSlope;
static constexpr float zKink = 1780.0f;

static constexpr float minMomentum =       1.5f * Gaudi::Units::GeV;
static constexpr float minPT =             0.3f * Gaudi::Units::GeV;
static constexpr float maxPseudoChi2 =     1280.0f;
static constexpr float yTol =              0.5f * Gaudi::Units::mm;
static constexpr float yTolSlope =         0.08f;
static constexpr float hitTol1 =           6.0f * Gaudi::Units::mm;
static constexpr float hitTol2 =           0.8f * Gaudi::Units::mm;
static constexpr float deltaTx1 =          0.035f;
static constexpr float deltaTx2 =          0.018f;
static constexpr float maxXSlope =         0.350f;
static constexpr float maxYSlope =         0.300f;
static constexpr float centralHoleSize =   33.0f * Gaudi::Units::mm;
static constexpr float intraLayerDist =    15.0f * Gaudi::Units::mm;
static constexpr float overlapTol =        0.7f* Gaudi::Units::mm;
static constexpr float passHoleSize =      40.0f * Gaudi::Units::mm;
static constexpr int   minHighThres =      1;
static constexpr bool  printVariables =    false;
static constexpr bool  passTracks =        false;
static constexpr bool  doTiming =          false;
// Scale the z-component, to not run into numerical problems with floats
// first add to sum values from hit at xMidField, zMidField hit
static constexpr float zDiff =             0.001f * (zKink - zMidUT);

} // namespace Constants
} // namespace UT

struct UTBoards {
  uint32_t number_of_boards;
  uint32_t number_of_channels;
  uint32_t* stripsPerHybrids;
  uint32_t* stations;
  uint32_t* layers;
  uint32_t* detRegions;
  uint32_t* sectors;
  uint32_t* chanIDs;

  UTBoards(const std::vector<char>& ut_boards);

  __device__ __host__ UTBoards(const char* ut_boards);
};

struct UTGeometry {
  uint32_t number_of_sectors;
  uint32_t* firstStrip;
  float* pitch;
  float* dy;
  float* dp0diX;
  float* dp0diY;
  float* dp0diZ;
  float* p0X;
  float* p0Y;
  float* p0Z;
  float* cos;

  UTGeometry(const std::vector<char>& ut_geometry);

  __device__ __host__ UTGeometry(const char* ut_geometry);
};
