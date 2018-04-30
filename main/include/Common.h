#pragma once

#include <vector>
#include <iostream>
#include <stdint.h>


// MC check on?
#ifdef MC_CHECK
  const bool do_mc_check = true;
 #else
  const bool do_mc_check = false;
#endif

/**
 * Generic StrException launcher
 */
class StrException : public std::exception
{
public:
    std::string s;
    StrException(std::string ss) : s(ss) {}
    ~StrException() throw () {} // Updated
    const char* what() const throw() { return s.c_str(); }
};

/**
 * @brief Struct to typecast events.
 */
struct EventInfo {
  size_t size;
  uint32_t numberOfModules;
  uint32_t numberOfHits;
  float* module_Zs;
  uint32_t* module_hitStarts;
  uint32_t* module_hitNums;
  uint32_t* hit_IDs;
  float* hit_Xs;
  float* hit_Ys;
  float* hit_Zs;

  EventInfo() = default;

  EventInfo(const std::vector<char>& event) {
    char* input = (char*) event.data();

    numberOfModules  = *((uint32_t*)input); input += sizeof(uint32_t);
    numberOfHits     = *((uint32_t*)input); input += sizeof(uint32_t);
    module_Zs        = (float*)input; input += sizeof(float) * numberOfModules;
    module_hitStarts = (uint32_t*)input; input += sizeof(uint32_t) * numberOfModules;
    module_hitNums   = (uint32_t*)input; input += sizeof(uint32_t) * numberOfModules;
    hit_IDs          = (uint32_t*)input; input += sizeof(uint32_t) * numberOfHits;
    hit_Xs           = (float*)  input; input += sizeof(float)   * numberOfHits;
    hit_Ys           = (float*)  input; input += sizeof(float)   * numberOfHits;
    hit_Zs           = (float*)  input; input += sizeof(float)   * numberOfHits;

    size = input - event.data();
  }

  EventInfo(const char* event) {
    char* input = (char*) event;
    
    numberOfModules  = *((uint32_t*)input); input += sizeof(uint32_t);
    numberOfHits     = *((uint32_t*)input); input += sizeof(uint32_t);
    module_Zs        = (float*)input; input += sizeof(float) * numberOfModules;
    module_hitStarts = (uint32_t*)input; input += sizeof(uint32_t) * numberOfModules;
    module_hitNums   = (uint32_t*)input; input += sizeof(uint32_t) * numberOfModules;
    hit_IDs          = (uint32_t*)input; input += sizeof(uint32_t) * numberOfHits;
    hit_Xs           = (float*)  input; input += sizeof(float)   * numberOfHits;
    hit_Ys           = (float*)  input; input += sizeof(float)   * numberOfHits;
    hit_Zs           = (float*)  input; input += sizeof(float)   * numberOfHits;

    size = input - (char*) event;
  }
};
