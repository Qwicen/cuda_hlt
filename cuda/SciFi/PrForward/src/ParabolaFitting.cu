#include "ParabolaFitting.cuh"


__host__ __device__ int fitParabola(
  int* coordToFit,
  const int n_coordToFit,
  const SciFi::SciFiHits& scifi_hits,
  float trackParameters[SciFi::Tracking::nTrackParams],
  const bool xFit ) {

  //== Fit a cubic
  float s0   = 0.f; 
  float sz   = 0.f; 
  float sz2  = 0.f; 
  float sz3  = 0.f; 
  float sz4  = 0.f; 
  float sd   = 0.f; 
  float sdz  = 0.f; 
  float sdz2 = 0.f; 
  
  for ( int i_hit = 0; i_hit < n_coordToFit; ++i_hit) {
    int hit = coordToFit[i_hit];
    float d = trackToHitDistance(trackParameters, scifi_hits, hit);
    if (!xFit)
      d *= - 1.f / scifi_hits.dxdy[hit];//TODO multiplication much faster than division!
    float w = scifi_hits.w[hit];
    float z = .001f * ( scifi_hits.z0[hit] - SciFi::Tracking::zReference );
    s0   += w;
    sz   += w * z; 
    sz2  += w * z * z; 
    sz3  += w * z * z * z; 
    sz4  += w * z * z * z * z; 
    sd   += w * d; 
    sdz  += w * d * z; 
    sdz2 += w * d * z * z; 
  }    
  const float b1 = sz  * sz  - s0  * sz2; 
  const float c1 = sz2 * sz  - s0  * sz3; 
  const float d1 = sd  * sz  - s0  * sdz; 
  const float b2 = sz2 * sz2 - sz * sz3; 
  const float c2 = sz3 * sz2 - sz * sz4; 
  const float d2 = sdz * sz2 - sz * sdz2;
  const float den = (b1 * c2 - b2 * c1 );
  if(!(fabsf(den) > 1e-5f)) return false;
  const float db  = (d1 * c2 - d2 * c1 ) / den; 
  const float dc  = (d2 * b1 - d1 * b2 ) / den; 
  const float da  = ( sd - db * sz - dc * sz2) / s0;
  if (xFit) {
    trackParameters[0] += da;
    trackParameters[1] += db*1.e-3f;
    trackParameters[2] += dc*1.e-6f;
  } else {
    trackParameters[4] += da;
    trackParameters[5] += db*1.e-3f;
    trackParameters[6] += dc*1.e-6f;
  }

  return true;
}
 
