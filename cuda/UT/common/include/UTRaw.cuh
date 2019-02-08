#pragma once

#include <stdint.h>

struct UTRawBank {
  uint32_t sourceID;
  uint32_t number_of_hits;
  uint16_t* data;

  __device__ __host__ UTRawBank(const char* ut_raw_bank)
  {
    uint32_t* p = (uint32_t*) ut_raw_bank;
    sourceID = *p;
    p += 1;
    number_of_hits = *p & 0x0000FFFFU;
    p += 1;
    data = (uint16_t*) p;
  }
};

struct UTRawEvent {
  uint32_t number_of_raw_banks;
  uint32_t* raw_bank_offsets;
  char* data;

  __device__ __host__ UTRawEvent(const char* event)
  {
    const char* p = event;
    number_of_raw_banks = *((uint32_t*) p);
    p += sizeof(uint32_t);
    raw_bank_offsets = (uint32_t*) p;
    p += (number_of_raw_banks + 1) * sizeof(uint32_t);
    data = (char*) p;
  }

  __device__ __host__ UTRawBank getUTRawBank(const uint32_t index) const
  {
    const uint32_t offset = raw_bank_offsets[index];
    UTRawBank raw_bank(data + offset);
    return raw_bank;
  }
};
