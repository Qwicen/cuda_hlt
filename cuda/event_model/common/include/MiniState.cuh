#pragma once

#include "VeloConsolidated.cuh"

struct MiniState {
  float x, y, z, tx, ty;

  __host__ __device__ MiniState() {};

  __host__ __device__ MiniState(const MiniState& other) : x(other.x), y(other.y), z(other.z), tx(other.tx), ty(other.ty)
  {}

  __host__ __device__ MiniState(const Velo::Consolidated::States& velo_states, const uint index) :
    x(velo_states.x[index]), y(velo_states.y[index]), tx(velo_states.tx[index]), ty(velo_states.ty[index]),
    z(velo_states.z[index])
  {}

  __host__ __device__ MiniState(const float _x, const float _y, const float _z, const float _tx, const float _ty) :
    x(_x), y(_y), z(_z), tx(_tx), ty(_ty)
  {}
};
