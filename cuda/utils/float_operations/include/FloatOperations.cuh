#pragma once

/*
  Operators for float2 and float3
  originally implemented by A. Kozlinskiy in the context of Mu3e
 */

// float2
__host__ __device__ inline float float2_dot(const float2& l, const float2& r) { return l.x * r.x + l.y * r.y; }

__host__ __device__ inline float2 operator+(const float2& l, const float2& r) { return {l.x + r.x, l.y + r.y}; }

__host__ __device__ inline float2 operator-(const float2& l, const float2& r) { return {l.x - r.x, l.y - r.y}; }

__host__ __device__ inline float2 operator*(const float2& l, float r) { return {l.x * r, l.y * r}; }

__host__ __device__ inline float2 operator/(const float2& l, float r) { return {l.x / r, l.y / r}; }

__host__ __device__ inline float2& operator+=(float2& l, const float2& r)
{
  l.x += r.x;
  l.y += r.y;
  return l;
}

__host__ __device__ inline float2& operator-=(float2& l, const float2& r)
{
  l.x -= r.x;
  l.y -= r.y;
  return l;
}

// float3
__host__ __device__ inline float3 operator+(const float3& l, const float3& r)
{
  return {l.x + r.x, l.y + r.y, l.z + r.z};
}

__host__ __device__ inline float3 operator-(const float3& l, const float3& r)
{
  return {l.x - r.x, l.y - r.y, l.z - r.z};
}

__host__ __device__ inline float3 operator*(const float3& l, float r) { return {l.x * r, l.y * r, l.z * r}; }

__host__ __device__ inline float3 operator/(const float3& l, float r) { return {l.x / r, l.y / r, l.z / r}; }

__host__ __device__ inline float3& operator+=(float3& l, const float3& r)
{
  l.x += r.x;
  l.y += r.y;
  l.z += r.z;
  return l;
}

__host__ __device__ inline float3& operator-=(float3& l, const float3& r)
{
  l.x -= r.x;
  l.y -= r.y;
  l.z -= r.z;
  return l;
}
