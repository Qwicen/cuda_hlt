#pragma once

#include "Argument.cuh"
#include "patPV_Definitions.cuh"
#include "TrackBeamLineVertexFinder.cuh"

/**
 * @brief Definition of arguments. All arguments should be defined here,
 *        with their associated type.
 */
ARGUMENT(dev_pvtracks, PVTrack)
ARGUMENT(dev_zhisto, float)
ARGUMENT(dev_zpeaks, float)
ARGUMENT(dev_number_of_zpeaks, uint)
ARGUMENT(dev_multi_fit_vertices, PV::Vertex)
ARGUMENT(dev_number_of_multi_fit_vertices, uint)
ARGUMENT(dev_seeds, PatPV::XYZPoint)
ARGUMENT(dev_number_seeds, uint)
ARGUMENT(dev_vertex, PV::Vertex)
ARGUMENT(dev_number_vertex, int)
