#include "../include/VeloUTDefinitions.cuh"

__host__ __device__ VeloUTTracking::Hit VeloUTTracking::createHit( HitsSoA *hits_layers, const int hit_index ) {
    Hit hit;
    hit.m_cos = hits_layers->cos(hit_index);
    hit.m_yBegin = hits_layers->yBegin(hit_index);
    hit.m_yEnd = hits_layers->yEnd(hit_index);
    hit.m_zAtYEq0 = hits_layers->zAtYEq0(hit_index);
    hit.m_xAtYEq0 = hits_layers->xAtYEq0(hit_index);
    hit.m_weight = hits_layers->weight(hit_index);
    hit.m_highThreshold = hits_layers->highThreshold(hit_index);
    hit.m_LHCbID = hits_layers->LHCbID(hit_index);
    hit.m_planeCode = hits_layers->planeCode(hit_index);
    
    return hit;
  }
