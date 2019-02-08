#pragma once

#include <stdint.h>

namespace SciFi {
  struct SciFiRawBank {
    uint32_t sourceID;
    uint16_t* data;
    uint16_t* last;

    __device__ __host__ SciFiRawBank(const char* raw_bank, const char* end)
    {
      const char* p = raw_bank;
      sourceID = *((uint32_t*) p);
      p += sizeof(uint32_t);
      data = (uint16_t*) p;
      last = (uint16_t*) end;
    }
  };

  struct SciFiRawEvent {
    uint32_t number_of_raw_banks;
    uint32_t* raw_bank_offset;
    char* payload;

    __device__ __host__ SciFiRawEvent(const char* event)
    {
      const char* p = event;
      number_of_raw_banks = *((uint32_t*) p);
      p += sizeof(uint32_t);
      raw_bank_offset = (uint32_t*) p;
      p += (number_of_raw_banks + 1) * sizeof(uint32_t);
      payload = (char*) p;
    }
    __device__ __host__ SciFiRawBank getSciFiRawBank(const uint32_t index) const
    {
      SciFiRawBank bank(payload + raw_bank_offset[index], payload + raw_bank_offset[index + 1]);
      return bank;
    }
  };

  namespace SciFiRawBankParams { // from SciFi/SciFiDAQ/src/SciFiRawBankParams.h
    enum shifts {
      linkShift = 9,
      cellShift = 2,
      fractionShift = 1,
      sizeShift = 0,
    };

    static constexpr uint16_t nbClusMaximum = 31;   // 5 bits
    static constexpr uint16_t nbClusFFMaximum = 10; //
    static constexpr uint16_t fractionMaximum = 1;  // 1 bits allocted
    static constexpr uint16_t cellMaximum = 127;    // 0 to 127; coded on 7 bits
    static constexpr uint16_t sizeMaximum = 1;      // 1 bits allocated

    enum BankProperties { NbBanks = 240, NbLinksPerBank = 24 };

    static constexpr uint16_t clusterMaxWidth = 4;
  } // namespace SciFiRawBankParams

} // namespace SciFi
