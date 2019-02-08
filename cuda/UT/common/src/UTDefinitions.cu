#include "UTDefinitions.cuh"

UTBoards::UTBoards(const std::vector<char>& ut_boards)
{
  uint32_t* p = (uint32_t*) ut_boards.data();
  number_of_boards = *p;
  p += 1;
  number_of_channels = UT::Decoding::ut_number_of_sectors_per_board * number_of_boards;
  stripsPerHybrids = p;
  p += number_of_channels;
  stations = p;
  p += number_of_channels;
  layers = p;
  p += number_of_channels;
  detRegions = p;
  p += number_of_channels;
  sectors = p;
  p += number_of_channels;
  chanIDs = p;
  p += number_of_channels;
}

__device__ __host__ UTBoards::UTBoards(const char* ut_boards)
{
  uint32_t* p = (uint32_t*) ut_boards;
  number_of_boards = *p;
  p += 1;
  number_of_channels = UT::Decoding::ut_number_of_sectors_per_board * number_of_boards;
  stripsPerHybrids = p;
  p += number_of_boards;
  stations = p;
  p += number_of_channels;
  layers = p;
  p += number_of_channels;
  detRegions = p;
  p += number_of_channels;
  sectors = p;
  p += number_of_channels;
  chanIDs = p;
  p += number_of_channels;
}

UTGeometry::UTGeometry(const std::vector<char>& ut_geometry)
{

  uint32_t* p = (uint32_t*) ut_geometry.data();
  number_of_sectors = *((uint32_t*) p);
  p += 1;
  firstStrip = (uint32_t*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
  pitch = (float*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
  dy = (float*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
  dp0diX = (float*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
  dp0diY = (float*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
  dp0diZ = (float*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
  p0X = (float*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
  p0Y = (float*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
  p0Z = (float*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
  cos = (float*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
}

__device__ __host__ UTGeometry::UTGeometry(const char* ut_geometry)
{
  uint32_t* p = (uint32_t*) ut_geometry;
  number_of_sectors = *((uint32_t*) p);
  p += 1;
  firstStrip = (uint32_t*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
  pitch = (float*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
  dy = (float*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
  dp0diX = (float*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
  dp0diY = (float*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
  dp0diZ = (float*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
  p0X = (float*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
  p0Y = (float*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
  p0Z = (float*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
  cos = (float*) p;
  p += UT::Decoding::ut_number_of_geometry_sectors;
}
