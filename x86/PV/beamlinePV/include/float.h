#pragma once

/*
  Define float2 and float3 for when CUDACC is not defined
  // originally implemented by A. Kozlinskiy in the context of Mu3e
 */

// float2
#ifndef __CUDACC__
struct float2 {
  float x = 0;
  float y = 0;

  float2() {}
float2(float x_, float y_) : x(x_), y(y_) {}

};
#endif // __CUDACC__

__host__ __device__
inline
float float2_dot(const float2& l, const float2& r) {
  return l.x * r.x + l.y * r.y;
}

__host__ __device__
inline
float2 operator + (const float2& l, const float2& r) {
  return { l.x + r.x, l.y + r.y };
}

__host__ __device__
inline
float2 operator - (const float2& l, const float2& r) {
  return { l.x - r.x, l.y - r.y };
}

__host__ __device__
inline
float2 operator * (const float2& l, float r) {
  return { l.x * r, l.y * r };
}

__host__ __device__
inline
float2 operator / (const float2& l, float r) {
  return { l.x / r, l.y / r };
}

__host__ __device__
inline
float2& operator += (float2& l, const float2& r) {
  l.x += r.x; l.y += r.y;
  return l;
}

__host__ __device__
inline
float2& operator -= (float2& l, const float2& r) {
  l.x -= r.x; l.y -= r.y;
  return l;
}

// float3

#ifndef __CUDACC__
struct float3 : float2 {
  float z = 0;

  float3() {}
 float3(float x_, float y_, float z_) : float2(x_, y_), z(z_) {}

};
#endif // __CUDACC__

__host__ __device__
inline
float3 operator + (const float3& l, const float3& r) {
  return { l.x + r.x, l.y + r.y, l.z + r.z };
}

__host__ __device__
inline
float3 operator - (const float3& l, const float3& r) {
  return { l.x - r.x, l.y - r.y, l.z - r.z };
}

__host__ __device__
inline
float3 operator * (const float3& l, float r) {
  return { l.x * r, l.y * r, l.z * r };
}

__host__ __device__
inline
float3 operator / (const float3& l, float r) {
  return { l.x / r, l.y / r, l.z / r };
}

__host__ __device__
inline
float3& operator += (float3& l, const float3& r) {
  l.x += r.x; l.y += r.y; l.z += r.z;
  return l;
}

__host__ __device__
inline
float3& operator -= (float3& l, const float3& r) {
  l.x -= r.x; l.y -= r.y; l.z -= r.z;
  return l;
}
