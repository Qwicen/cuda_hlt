#include "../include/VeloUTDefinitions.cuh"

__constant__ float VeloUTTracking::dev_dxDyTable[VeloUTTracking::n_layers];

__host__ __device__ VeloUTTracking::Hit VeloUTTracking::createHit( HitsSoA *hits_layers, const int i_layer, const int i_hit ) {
    Hit hit;
    const int layer_offset = hits_layers->layer_offset[i_layer];
    hit.m_cos = hits_layers->cos(layer_offset + i_hit);
    hit.m_yBegin = hits_layers->yBegin(layer_offset + i_hit);
    hit.m_yEnd = hits_layers->yEnd(layer_offset + i_hit);
    hit.m_zAtYEq0 = hits_layers->zAtYEq0(layer_offset + i_hit);
    hit.m_xAtYEq0 = hits_layers->xAtYEq0(layer_offset + i_hit);
    hit.m_weight = hits_layers->weight(layer_offset + i_hit);
    hit.m_highThreshold = hits_layers->highThreshold(layer_offset + i_hit);
    hit.m_LHCbID = hits_layers->LHCbID(layer_offset + i_hit);
    hit.m_planeCode = hits_layers->planeCode(layer_offset + i_hit);
    
    return hit;
  }
