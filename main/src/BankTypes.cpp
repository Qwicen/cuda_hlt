#include <map>
#include <string>
#include <BankTypes.h>

namespace {
   const std::map<BankTypes, std::string> BankNames = {{BankTypes::VP, "VP"},
                                                       {BankTypes::UT, "UT"},
                                                       {BankTypes::FT, "FTCluster"},
                                                       {BankTypes::MUON, "Muon"}};
}

std::string bank_name(BankTypes type)
{
   auto it = BankNames.find(type);
   if (it != end(BankNames)) {
      return it->second;
   } else {
      return "Unknown";
   }
}
