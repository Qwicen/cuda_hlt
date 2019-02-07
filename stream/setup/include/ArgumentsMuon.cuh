#pragma once

#include "Argument.cuh"
#include "MuonDefinitions.cuh"

/**
 * @brief Definition of arguments. All arguments should be defined here,
 *        with their associated type.
 */
ARGUMENT(dev_muon_hits, Muon::HitsSoA)
ARGUMENT(dev_muon_catboost_features, float)
ARGUMENT(dev_muon_catboost_output, float)
