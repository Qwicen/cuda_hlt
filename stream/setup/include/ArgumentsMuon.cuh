#pragma once

#include "Argument.cuh"
#include "ArgumentsCommon.cuh"
#include "MuonDefinitions.cuh"

/**
 * @brief Definition of arguments. All arguments should be defined here,
 *        with their associated type.
 */
ARGUMENT(dev_muon_hits, Muon::HitsSoA)
ARGUMENT(dev_muon_track_occupancies, int)
ARGUMENT(dev_is_muon, bool)
ARGUMENT(dev_muon_catboost_features, float)
ARGUMENT(dev_muon_catboost_output, float)
