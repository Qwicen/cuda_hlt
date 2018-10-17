#include "UTDefinitions.cuh"

UTBoards::UTBoards(const std::vector<char> & ut_boards) {
  uint32_t * p = (uint32_t *) ut_boards.data();
  number_of_boards   = *p; p += 1;
  number_of_channels = UTDecoding::ut_number_of_sectors_per_board * number_of_boards;
  stripsPerHybrids   = p;  p += number_of_channels;
  stations           = p;  p += number_of_channels;
  layers             = p;  p += number_of_channels;
  detRegions         = p;  p += number_of_channels;
  sectors            = p;  p += number_of_channels;
  chanIDs            = p;  p += number_of_channels;
}

__device__ __host__ UTBoards::UTBoards(
    const char * ut_boards
) {
  uint32_t * p = (uint32_t *) ut_boards;
  number_of_boards   = *p; p += 1;
  number_of_channels = UTDecoding::ut_number_of_sectors_per_board * number_of_boards;
  stripsPerHybrids   = p;  p += number_of_boards;
  stations           = p;  p += number_of_channels;
  layers             = p;  p += number_of_channels;
  detRegions         = p;  p += number_of_channels;
  sectors            = p;  p += number_of_channels;
  chanIDs            = p;  p += number_of_channels;
}

UTGeometry::UTGeometry(const std::vector<char> & ut_geometry) {

  uint32_t * p = (uint32_t *) ut_geometry.data();
  number_of_sectors = *((uint32_t *)p); p += 1;
  firstStrip = (uint32_t *)p; p += UTDecoding::ut_number_of_geometry_sectors;
  pitch      = (float *)   p; p += UTDecoding::ut_number_of_geometry_sectors;
  dy         = (float *)   p; p += UTDecoding::ut_number_of_geometry_sectors;
  dp0diX     = (float *)   p; p += UTDecoding::ut_number_of_geometry_sectors;
  dp0diY     = (float *)   p; p += UTDecoding::ut_number_of_geometry_sectors;
  dp0diZ     = (float *)   p; p += UTDecoding::ut_number_of_geometry_sectors;
  p0X        = (float *)   p; p += UTDecoding::ut_number_of_geometry_sectors;
  p0Y        = (float *)   p; p += UTDecoding::ut_number_of_geometry_sectors;
  p0Z        = (float *)   p; p += UTDecoding::ut_number_of_geometry_sectors;
  cos        = (float *)   p; p += UTDecoding::ut_number_of_geometry_sectors;
}

__device__ __host__ UTGeometry::UTGeometry(
    const char * ut_geometry
) {
  uint32_t * p = (uint32_t *)ut_geometry;
  number_of_sectors = *((uint32_t *)p); p +=1;
  firstStrip = (uint32_t *)p; p += UTDecoding::ut_number_of_geometry_sectors;
  pitch      = (float *)   p; p += UTDecoding::ut_number_of_geometry_sectors;
  dy         = (float *)   p; p += UTDecoding::ut_number_of_geometry_sectors;
  dp0diX     = (float *)   p; p += UTDecoding::ut_number_of_geometry_sectors;
  dp0diY     = (float *)   p; p += UTDecoding::ut_number_of_geometry_sectors;
  dp0diZ     = (float *)   p; p += UTDecoding::ut_number_of_geometry_sectors;
  p0X        = (float *)   p; p += UTDecoding::ut_number_of_geometry_sectors;
  p0Y        = (float *)   p; p += UTDecoding::ut_number_of_geometry_sectors;
  p0Z        = (float *)   p; p += UTDecoding::ut_number_of_geometry_sectors;
  cos        = (float *)   p; p += UTDecoding::ut_number_of_geometry_sectors;
}

__device__ __host__ UTRawBank::UTRawBank (
  const char* ut_raw_bank
) {
  uint32_t* p = (uint32_t*) ut_raw_bank;
  sourceID       = *p;               p+=1;
  number_of_hits = *p & 0x0000FFFFU; p+=1;
  data           = (uint16_t*)p;
}

__device__ __host__ UTRawEvent::UTRawEvent (
  const uint32_t * ut_raw_event
) {
  uint32_t* p = (uint32_t *) ut_raw_event;
  number_of_raw_banks = *p; p += 1;
  raw_bank_offsets    =  p; p += (number_of_raw_banks + 1);
  data                =  (char*) p;
}

__device__ __host__
UTRawBank UTRawEvent::getUTRawBank (
  const uint32_t index
) const {
  const uint32_t offset = raw_bank_offsets[index];
  UTRawBank raw_bank(data + offset);
  return raw_bank;
}

UTHit::UTHit(float    ut_yBegin,
             float    ut_yEnd,
             float    ut_zAtYEq0,
             float    ut_xAtYEq0,
             float    ut_weight,
             uint32_t ut_highThreshold,
             uint32_t ut_LHCbID,
             uint32_t ut_planeCode
             ) {
  yBegin        = ut_yBegin       ;
  yEnd          = ut_yEnd         ;
  zAtYEq0       = ut_zAtYEq0      ;
  xAtYEq0       = ut_xAtYEq0      ;
  weight        = ut_weight       ;
  highThreshold = ut_highThreshold;
  LHCbID        = ut_LHCbID       ;
  planeCode     = ut_planeCode    ;
}

UTHits::UTHits(uint32_t* base_pointer, uint32_t total_number_of_hits) {
  raw_bank_index = base_pointer;
  yBegin = reinterpret_cast<float*>(base_pointer + total_number_of_hits);
  yEnd = reinterpret_cast<float*>(base_pointer + 2*total_number_of_hits);
  zAtYEq0 = reinterpret_cast<float*>(base_pointer + 3*total_number_of_hits);
  xAtYEq0 = reinterpret_cast<float*>(base_pointer + 4*total_number_of_hits);
  weight = reinterpret_cast<float*>(base_pointer + 5*total_number_of_hits);
  highThreshold = base_pointer + 6*total_number_of_hits;
  LHCbID = base_pointer + 7*total_number_of_hits;
  planeCode = base_pointer + 8*total_number_of_hits;
}

UTHit UTHits::getHit(uint32_t index) const {
  return {yBegin[index],
   yEnd[index],
   zAtYEq0[index],
   xAtYEq0[index],
   weight[index],
   highThreshold[index],
   LHCbID[index],
   planeCode[index]
 };
}
