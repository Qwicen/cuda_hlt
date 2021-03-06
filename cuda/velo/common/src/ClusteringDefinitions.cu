#include "ClusteringDefinitions.cuh"

__device__ __host__ VeloRawEvent::VeloRawEvent(const char* event)
{
  const char* p = event;
  number_of_raw_banks = *((uint32_t*) p);
  p += sizeof(uint32_t);
  raw_bank_offset = (uint32_t*) p;
  p += (number_of_raw_banks + 1) * sizeof(uint32_t);
  payload = (char*) p;
}

__device__ __host__ VeloRawBank::VeloRawBank(const char* raw_bank)
{
  const char* p = raw_bank;
  sensor_index = *((uint32_t*) p);
  p += sizeof(uint32_t);
  sp_count = *((uint32_t*) p);
  p += sizeof(uint32_t);
  sp_word = (uint32_t*) p;
}

VeloGeometry::VeloGeometry(const std::vector<char>& geometry)
{
  const char* p = geometry.data();

  local_x = (float*) p;
  p += sizeof(float) * Velo::Constants::number_of_sensor_columns;
  x_pitch = (float*) p;
  p += sizeof(float) * Velo::Constants::number_of_sensor_columns;
  ltg = (float*) p;
  p += sizeof(float) * 16 * Velo::Constants::n_sensors;

  size = p - geometry.data();

  if (size != geometry.size()) {
    error_cout << "Size mismatch for geometry" << std::endl;
  }
}

__device__ __host__ VeloGeometry::VeloGeometry(const char* geometry)
{
  const char* p = geometry;

  local_x = (float*) p;
  p += sizeof(float) * Velo::Constants::number_of_sensor_columns;
  x_pitch = (float*) p;
  p += sizeof(float) * Velo::Constants::number_of_sensor_columns;
  ltg = (float*) p;
  p += sizeof(float) * 16 * Velo::Constants::n_sensors;

  size = p - geometry;
}

__device__ __host__ uint32_t get_channel_id(unsigned int sensor, unsigned int chip, unsigned int col, unsigned int row)
{
  return (sensor << LHCb::VPChannelID::sensorBits) | (chip << LHCb::VPChannelID::chipBits) |
         (col << LHCb::VPChannelID::colBits) | row;
}

__device__ __host__ int32_t get_lhcb_id(int32_t cid) { return (LHCb::VPChannelID::VP << LHCb::detectorTypeBits) + cid; }
