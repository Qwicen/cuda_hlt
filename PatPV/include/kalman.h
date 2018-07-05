








inline data_t filter(const data_t z, data_t &x, data_t &tx, data_t &covXX,
          data_t &covXTx, data_t &covTxTx, const data_t zhit, 
          const data_t xhit, const data_t whit) {

  // compute the prediction
  const data_t dz = zhit - z;
  const data_t predx = x + dz * tx;
   
  const data_t dz_t_covTxTx = dz * covTxTx;
  const data_t predcovXTx = covXTx + dz_t_covTxTx;
  const data_t dx_t_covXTx = dz * covXTx;
  
  const data_t predcovXX = covXX + 2 * dx_t_covXTx + dz * dz_t_covTxTx;
  const data_t predcovTxTx = covTxTx;
  // compute the gain matrix
  const data_t R = 1.0 / (1.0 / whit + predcovXX);
  const data_t Kx = predcovXX * R;
  const data_t KTx = predcovXTx * R;
  // update the state vector
  const data_t r = xhit - predx;
  x = predx + Kx * r;
  tx = tx + KTx * r;


  // update the covariance matrix. we can write it in many ways ...
  covXX /*= predcovXX  - Kx * predcovXX */ = (1 - Kx) * predcovXX;
  covXTx /*= predcovXTx - predcovXX * predcovXTx / R */ = (1 - Kx) * predcovXTx;
  covTxTx = predcovTxTx - KTx * predcovXTx;
  
  //printf("predcovXX = %.f, R = %f, Kx = %.10f, KTx = %f, 1-Kx = %.10f \n ", predcovXX, R, Kx, KTx, 1-Kx);

  // return the chi2
  return r * r * R;
}



/* Apply Kalman filter to a track with all hits known already  */
void fitKalman( track_t &tr, state_t &state, const int direction ) {

  int firsthit = 0; // 0;
  int dhit = +1;
  
  int lasthit;

  lasthit = tr.hits.size() - 1;
  if ( ( tr.hits[lasthit]->z -tr.hits[firsthit]->z ) * direction < 0 ) {
    std::swap(firsthit, lasthit);
    dhit = -1;
  }

  // first track state
  hit_t hit, hit1;
  hit  = *(tr.hits[firsthit]);
  hit1 = *(tr.hits[firsthit + dhit]);


  data_t x = hit.x;
  data_t y = hit.y;
  data_t z = hit.z;
  data_t dz = hit1.z - z;
  
  // use slope between first two hits as initial direction
  data_t dx = hit1.x - x;
  data_t tx = dx / dz;
  data_t ty = (hit1.y - y) / dz;
 
  // initialize the covariance matrix
  // Pierre's comment: should be initially (first filter step) much larger:
  // at least 100 * the actual error 
  data_t covXX = 1. / (WEIGHT);
  data_t covYY = 1. / (WEIGHT);
 
  data_t covXTx = 0.; // no initial correlation
  data_t covYTy = 0.; 
  data_t covTxTx = 1.;  // randomly large error
  data_t covTyTy = 1.;
   
  // add remaining hits
  data_t chi2 = 0;
  data_t chi2_x = 0;
  data_t chi2_y = 0;
  for ( int i = firsthit + dhit; i != lasthit + dhit; i += dhit ) {
    //int i = firsthit + dhit;
    hit = *(tr.hits[i]);
    // add the noise
    // Parameters for kalmanfit scattering. calibrated on MC, shamelessly hardcoded:
    // (Taken from PrPixelTracking.cpp)
    data_t scat2 = FACTOR * (1e-8 + 7e-6 * (tx * tx + ty * ty) );
    covTxTx += scat2;
    covTyTy += scat2;
    
/* #ifndef __CUDACC__ */
/*     tr.hits[i-dhit]->scat2 = scat2; */
/* #endif */
     
    // filter X
    data_t return_chi2 = filter( z, x, tx, covXX, covXTx, covTxTx, hit.z, hit.x, WEIGHT ); 
    chi2 += return_chi2;
    chi2_x += return_chi2;
    // filter Y
    return_chi2 = filter( z, y, ty, covYY, covYTy, covTyTy, hit.z, hit.y, WEIGHT );
    chi2 += return_chi2;
    chi2_y += return_chi2;
    // update z (not done in the filter, since needed only once)
    z = hit.z;
  }
     
  // finally, fill the state
  state.x = x; 
  state.y = y;
  state.z = z;
  state.tx = tx;
  state.ty = ty;
  state.covXX = covXX;
  state.covXTx = covXTx;
  state.covTxTx = covTxTx;
  state.covYY = covYY;
  state.covYTy = covYTy;
  state.covTyTy = covTyTy;
  state.chi2 = chi2;
  state.chi2_x = chi2_x;
  state.chi2_y = chi2_y; 
}


data_t finalizeKalman( state_t &state ) {
  
  // add noise at last hit
  data_t scat2 = FACTOR * ( 1e-8 + 7e-6 * (state.tx *state. tx + state.ty * state.ty) );
  state.covTxTx += scat2;
  state.covTyTy += scat2;

  return scat2;
}


