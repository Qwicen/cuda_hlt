#ifndef READ_MDF_H
#define READ_MDF_H 1

#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <BankTypes.h>

#include <gsl-lite.hpp>

#include "odin.hpp"
#include "raw_bank.hpp"
#include "mdf_header.hpp"

namespace LHCbToGPU {
   const std::unordered_map<LHCb::RawBank::BankType, BankTypes> bank_types =
      {{LHCb::RawBank::VP, BankTypes::VP},
       {LHCb::RawBank::UT, BankTypes::UT},
       {LHCb::RawBank::FTCluster, BankTypes::FT},
       {LHCb::RawBank::Muon, BankTypes::MUON}};

   using buffer_map = std::unordered_map<BankTypes, std::pair<std::vector<char>,
                                                              std::vector<unsigned int>>>;
}

namespace MDF {
void dump_hex(const char* start, int size);

std::tuple<bool, bool, gsl::span<char>> read_event(std::ifstream& input, LHCb::MDFHeader& h,
                                                   std::vector<char>& buffer, bool dbg = false);

std::tuple<bool, bool, gsl::span<char>> read_banks(std::ifstream& input, const LHCb::MDFHeader& h,
                                                   std::vector<char>& buffer, bool dbg = false);

std::tuple<size_t, LHCbToGPU::buffer_map, std::vector<LHCb::ODIN>>
read_events(size_t n, const std::vector<std::string>& files,
            const std::unordered_set<BankTypes>& types,
            size_t offset = 0);

LHCb::ODIN decode_odin(const LHCb::RawBank* bank);

}
#endif
