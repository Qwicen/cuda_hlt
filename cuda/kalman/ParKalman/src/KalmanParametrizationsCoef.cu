#include <stdio.h>
#include "KalmanParametrizationsCoef.cuh"

namespace ParKalmanFilter {

  __device__ __host__ StandardCoefs operator+(const StandardCoefs& a, const StandardCoefs& b)
  {
    StandardCoefs c;
    int nMax = 4 * a.degx1 + 2 * a.degx2 + 4 * a.degy1 + 2 * a.degy2;
    for (int i = 0; i < nMax; i++) {
      c.coefs[i] = a.coefs[i] + b.coefs[i];
    }
    return c;
  }

  __device__ __host__ StandardCoefs operator-(const StandardCoefs& a, const StandardCoefs& b)
  {
    StandardCoefs c;
    int nMax = 4 * a.degx1 + 2 * a.degx2 + 4 * a.degy1 + 2 * a.degy2;
    for (int i = 0; i < nMax; i++) {
      c.coefs[i] = a.coefs[i] - b.coefs[i];
    }
    return c;
  }

  __device__ __host__ StandardCoefs operator*(const StandardCoefs& a, double p)
  {
    StandardCoefs c;
    int nMax = 4 * a.degx1 + 2 * a.degx2 + 4 * a.degy1 + 2 * a.degy2;
    for (int i = 0; i < nMax; i++) {
      c.coefs[i] = p * a.coefs[i];
    }
    return c;
  }

} // namespace ParKalmanFilter
