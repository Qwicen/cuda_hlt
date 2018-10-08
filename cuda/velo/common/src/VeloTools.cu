#include "VeloTools.cuh"

__device__ float hit_phi_odd(
  const float x,
  const float y
) {
  return atan2(y, x);
}

__device__ float hit_phi_even(
  const float x,
  const float y
) {
  const auto phi = atan2(y, x);
  const auto less_than_zero = phi < 0.f;
  return phi + less_than_zero*2*CUDART_PI_F;
}
