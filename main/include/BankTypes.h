#ifndef BANKTYPES_H
#define BANKTYPES_H 1

#include <type_traits>
#include <iostream>
#include <unordered_map>
#include <gsl-lite.hpp>

namespace {
   using gsl::span;
}

namespace {
   constexpr auto BANKTYPES_START = __LINE__;
}
enum class BankTypes {
           VP,
           UT,
           FT,
           MUON
};
constexpr auto NBankTypes = __LINE__ - BANKTYPES_START - 4;
const std::unordered_map<BankTypes, float> BankSizes = {{BankTypes::VP, 51.77f},
                                                        {BankTypes::UT, 31.38f},
                                                        {BankTypes::FT, 54.47f},
                                                        {BankTypes::MUON, 5.13f}};


std::string bank_name(BankTypes type);

template<typename ENUM>
constexpr auto to_integral(ENUM e) -> typename std::underlying_type<ENUM>::type
{
   return static_cast<typename std::underlying_type<ENUM>::type>(e);
}

#endif
