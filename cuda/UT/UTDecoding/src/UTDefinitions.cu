#include "UTDefinitions.cuh"

UTBoards::UTBoards(const std::vector<char> & ut_boards) {
  uint32_t * p = (uint32_t *) ut_boards.data();
  number_of_boards   = *p; p += 1;
  number_of_channels = ut_number_of_sectors_per_board * number_of_boards;
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
  number_of_channels = ut_number_of_sectors_per_board * number_of_boards;
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
  firstStrip = (uint32_t *)p; p += ut_number_of_geometry_sectors;
  pitch      = (float *)   p; p += ut_number_of_geometry_sectors;
  dy         = (float *)   p; p += ut_number_of_geometry_sectors;
  dp0diX     = (float *)   p; p += ut_number_of_geometry_sectors;
  dp0diY     = (float *)   p; p += ut_number_of_geometry_sectors;
  dp0diZ     = (float *)   p; p += ut_number_of_geometry_sectors;
  p0X        = (float *)   p; p += ut_number_of_geometry_sectors;
  p0Y        = (float *)   p; p += ut_number_of_geometry_sectors;
  p0Z        = (float *)   p; p += ut_number_of_geometry_sectors;
  cos        = (float *)   p; p += ut_number_of_geometry_sectors;
}

__device__ __host__ UTGeometry::UTGeometry(
    const char * ut_geometry
) {
  uint32_t * p = (uint32_t *)ut_geometry;
  number_of_sectors = *((uint32_t *)p); p +=1;
  firstStrip = (uint32_t *)p; p += ut_number_of_geometry_sectors;
  pitch      = (float *)   p; p += ut_number_of_geometry_sectors;
  dy         = (float *)   p; p += ut_number_of_geometry_sectors;
  dp0diX     = (float *)   p; p += ut_number_of_geometry_sectors;
  dp0diY     = (float *)   p; p += ut_number_of_geometry_sectors;
  dp0diZ     = (float *)   p; p += ut_number_of_geometry_sectors;
  p0X        = (float *)   p; p += ut_number_of_geometry_sectors;
  p0Y        = (float *)   p; p += ut_number_of_geometry_sectors;
  p0Z        = (float *)   p; p += ut_number_of_geometry_sectors;
  cos        = (float *)   p; p += ut_number_of_geometry_sectors;
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
  raw_bank_offsets    =  p; p += number_of_raw_banks;
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

UTHit::UTHit(float    ut_cos,
             float    ut_yBegin,
             float    ut_yEnd,
             float    ut_zAtYEq0,
             float    ut_xAtYEq0,
             float    ut_weight,
             uint32_t ut_highThreshold,
             uint32_t ut_LHCbID,
             uint32_t ut_planeCode
             ) {
  cos           = ut_cos          ;
  yBegin        = ut_yBegin       ;
  yEnd          = ut_yEnd         ;
  zAtYEq0       = ut_zAtYEq0      ;
  xAtYEq0       = ut_xAtYEq0      ;
  weight        = ut_weight       ;
  highThreshold = ut_highThreshold;
  LHCbID        = ut_LHCbID       ;
  planeCode     = ut_planeCode    ;
}

void UTHits::typecast_unsorted(uint32_t* base_pointer, uint32_t total_number_of_hits) {
  m_cos = reinterpret_cast<float*>(base_pointer);
  m_yBegin = reinterpret_cast<float*>(base_pointer + total_number_of_hits);
  m_yEnd = reinterpret_cast<float*>(base_pointer + 2*total_number_of_hits);
  m_zAtYEq0 = reinterpret_cast<float*>(base_pointer + 3*total_number_of_hits);
  m_xAtYEq0 = reinterpret_cast<float*>(base_pointer + 4*total_number_of_hits);
  m_weight = reinterpret_cast<float*>(base_pointer + 5*total_number_of_hits);
  m_highThreshold = base_pointer + 6*total_number_of_hits;
  m_LHCbID = base_pointer + 7*total_number_of_hits;
  m_planeCode = base_pointer + 8*total_number_of_hits;
  m_temp = base_pointer + 9*total_number_of_hits;
}

void UTHits::typecast_sorted(uint32_t* base_pointer, uint32_t total_number_of_hits) {
  m_temp = base_pointer;
  m_cos = reinterpret_cast<float*>(base_pointer + total_number_of_hits);
  m_yBegin = reinterpret_cast<float*>(base_pointer + 2*total_number_of_hits);
  m_yEnd = reinterpret_cast<float*>(base_pointer + 3*total_number_of_hits);
  m_zAtYEq0 = reinterpret_cast<float*>(base_pointer + 4*total_number_of_hits);
  m_xAtYEq0 = reinterpret_cast<float*>(base_pointer + 5*total_number_of_hits);
  m_weight = reinterpret_cast<float*>(base_pointer + 6*total_number_of_hits);
  m_highThreshold = base_pointer + 7*total_number_of_hits;
  m_LHCbID = base_pointer + 8*total_number_of_hits;
  m_planeCode = base_pointer + 9*total_number_of_hits;
}

UTHit UTHits::getHit(uint32_t index) const {
  const float cos              = m_cos          [index];
  const float yBegin           = m_yBegin       [index];
  const float yEnd             = m_yEnd         [index];
  const float zAtYEq0          = m_zAtYEq0      [index];
  const float xAtYEq0          = m_xAtYEq0      [index];
  const float weight           = m_weight       [index];
  const uint32_t highThreshold = m_highThreshold[index];
  const uint32_t LHCbID        = m_LHCbID       [index];
  const uint32_t planeCode     = m_planeCode    [index];

  return UTHit(cos,
               yBegin,
               yEnd,
               zAtYEq0,
               xAtYEq0,
               weight,
               highThreshold,
               LHCbID,
               planeCode
               );
}

