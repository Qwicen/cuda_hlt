#include "LinearFitting.h"

/**
   Functions related to fitting a straight line
 */

float getLineFitDistance(ForwardTracking::LineFitterPars &parameters, ForwardTracking::HitsSoAFwd* hits_layers, int it )
{ 
  return hits_layers->m_coord[it] - (parameters.m_c0 + (hits_layers->m_z[it] - parameters.m_z0) * parameters.m_tc);
}

float getLineFitChi2(ForwardTracking::LineFitterPars &parameters, ForwardTracking::HitsSoAFwd* hits_layers, int it) {
  float d = getLineFitDistance( parameters, hits_layers, it ); 
  return d * d * hits_layers->m_coord[it]; 
}
void solveLineFit(ForwardTracking::LineFitterPars &parameters)  {
  float den = (parameters.m_sz*parameters.m_sz-parameters.m_s0*parameters.m_sz2);
  parameters.m_c0  = (parameters.m_scz * parameters.m_sz - parameters.m_sc * parameters.m_sz2) / den;
  parameters.m_tc  = (parameters.m_sc *  parameters.m_sz - parameters.m_s0 * parameters.m_scz) / den;
}

void incrementLineFitParameters(ForwardTracking::LineFitterPars &parameters, ForwardTracking::HitsSoAFwd* hits_layers, int it)
{
    float c = hits_layers->m_coord[it];
    float w = hits_layers->m_w[it];
    float z = hits_layers->m_z[it] - parameters.m_z0;
    parameters.m_s0   += w;
    parameters.m_sz   += w * z;
    parameters.m_sz2  += w * z * z;
    parameters.m_sc   += w * c;
    parameters.m_scz  += w * c * z;
} 

void fastLinearFit(
  ForwardTracking::HitsSoAFwd* hits_layers,
  std::vector<float> &trackParameters, 
  std::vector<unsigned int> &pc,
  int planelist[],
  PrParameters& pars)
{
  bool fit = true;
  while (fit) {
    //== Fit a line
    float s0   = 0.;
    float sz   = 0.;
    float sz2  = 0.;
    float sd   = 0.;
    float sdz  = 0.;

    for (auto hit : pc ){
      const float parsX[4] = {trackParameters[0],
                              trackParameters[1],
                              trackParameters[2],
                              trackParameters[3]};
      const float zHit = hits_layers->m_z[hit];
      float track_x_at_zHit = straightLineExtend(parsX,zHit);
      const float d = hits_layers->m_x[hit] - track_x_at_zHit;
      const float w = hits_layers->m_w[hit];
      const float z = zHit - Forward::zReference;
      s0   += w;
      sz   += w * z; 
      sz2  += w * z * z; 
      sd   += w * d; 
      sdz  += w * d * z; 
    }    
    float den = (sz*sz-s0*sz2);
    if( !(std::fabs(den) > 1e-5))return;
    const float da  = (sdz * sz - sd * sz2) / den; 
    const float db  = (sd *  sz - s0 * sdz) / den; 
    trackParameters[0] += da;
    trackParameters[1] += db;
    fit = false;

    if ( pc.size() < pars.minXHits ) return;

    int worst = pc.back();
    float maxChi2 = 0.f; 
    const bool notMultiple = nbDifferent(planelist) == pc.size();
    //TODO how many multiple hits do we normaly have?
    //how often do we do the right thing here?
    //delete two hits at same time?
    for ( auto hit : pc) { 
      // This could certainly be wrapped in some helper function with a lot
      // of passing around or copying etc... 
      const float parsX[4] = {trackParameters[0],
                              trackParameters[1],
                              trackParameters[2],
                              trackParameters[3]};
      float track_x_at_zHit = straightLineExtend(parsX,hits_layers->m_z[hit]);
      float hitdist = hits_layers->m_x[hit] - track_x_at_zHit; 
      float chi2 = hitdist*hitdist*hits_layers->m_w[hit];

      if ( chi2 > maxChi2 && ( notMultiple || planelist[hits_layers->m_planeCode[hit]/2] > 1 ) ) {
        maxChi2 = chi2;
        worst   = hit; 
      }    
    }    
    //== Remove grossly out hit, or worst in multiple layers
    if ( maxChi2 > Forward::maxChi2LinearFit || ( !notMultiple && maxChi2 > 4.f ) ) {
      planelist[hits_layers->m_planeCode[worst]/2] -= 1;
      std::vector<unsigned int> pc_temp;
      pc_temp.clear();
      for (auto hit : pc) {
        if (hit != worst) pc_temp.push_back(hit);
      }
      pc = pc_temp; 
    }
  }
}
 
