#pragma once

#include <tuple>

// All includes of all algorithms
#include "PrefixSum.cuh"
#include "PrefixSumHandler.cuh"
#include "InitEventList.cuh"
#include "GlobalEventCut.cuh"
#include "EstimateInputSize.cuh"
#include "MaskedVeloClustering.cuh"
#include "CalculatePhiAndSort.cuh"
#include "SearchByTriplet.cuh"
#include "ConsolidateVelo.cuh"
#include "UTCalculateNumberOfHits.cuh"
#include "UTPreDecode.cuh"
#include "UTFindPermutation.cuh"
#include "UTDecodeRawBanksInOrder.cuh"
#include "VeloUT.cuh"
#include "VeloEventModel.cuh"
#include "ConsolidateUT.cuh"
#include "SciFiCalculateClusterCount.cuh"
#include "SciFiPreDecode.cuh"
#include "SciFiRawBankDecoder.cuh"
#include "SciFiCalculateClusterCountV4.cuh"
#include "SciFiDirectDecoderV4.cuh"
#include "SciFiPreDecodeV4.cuh"
#include "SciFiRawBankDecoderV4.cuh"
#include "ConsolidateSciFi.cuh"
#include "SearchWindows.cuh"
#include "CompassUT.cuh"
#include "PrForward.cuh"
#include "VeloKalmanFilter.cuh"
#include "GetSeeds.cuh"
#include "FitSeeds.cuh"
#include "pv_beamline_extrapolate.cuh"
#include "pv_beamline_histo.cuh"
#include "pv_beamline_peak.cuh"
#include "pv_beamline_multi_fitter.cuh"
#include "RunForwardCPU.h"
#include "IPCut.cuh"
#include "VeloPVIP.cuh"
#include "RunBeamlinePVOnCPU.h"
#include "IsMuon.cuh"
#include "MuonFeaturesExtraction.cuh"
#include "MuonCatboostEvaluator.cuh"
#include "ParKalmanFilter.cuh"

#define SEQUENCE_T(...) typedef std::tuple<__VA_ARGS__> configured_sequence_t;

// SEQUENCE must be defined at compile time.
// Values passed at compile time should match
// the name of the file in "sequences/<filename>.cuh":
//
// "cmake -DSEQUENCE=<sequence_name> .." matches "sequences/<sequence_name>.cuh"
//
// eg.
// "cmake -DSEQUENCE=DefaultSequence .." (or just "cmake ..") matches "sequences/DefaultSequence.cuh"
// "cmake -DSEQUENCE=Velo .." matches "sequences/Velo.cuh"
// "cmake -DSEQUENCE=VeloUT .." matches "sequences/VeloUT.cuh"

#include "ConfiguredSequence.h"
